# artevolve_core.py
# Core GA logic and rendering for ArtEvolve (stability + exploration improvements)

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io

# ---------------------------
# Parameters defaults (can be overridden in app)
# ---------------------------
DEFAULT_NUM_SHAPES = 12
DEFAULT_IMAGE_SIZE = 300

# ---------------------------
# Population helper
# ---------------------------
def create_population(size, num_shapes=DEFAULT_NUM_SHAPES):
    """Utility so app.py can initialize populations easily."""
    return [random_genome(num_shapes) for _ in range(size)]

# ---------------------------
# Genome utilities
# ---------------------------
def random_gene():
    """One shape gene: [stype, x, y, sx, sy, angle, r, g, b, a]"""
    return [
        random.choice([0, 1]),           # shape type (reserved)
        random.random(),                 # x (0..1)
        random.random(),                 # y (0..1)
        random.random()*0.20 + 0.02,     # sx
        random.random()*0.20 + 0.02,     # sy
        random.random()*360.0,           # angle
        random.randint(0, 255),          # r
        random.randint(0, 255),          # g
        random.randint(0, 255),          # b
        random.random()*0.75 + 0.05      # alpha (0.05..0.8)
    ]

def random_genome(num_shapes=DEFAULT_NUM_SHAPES):
    return [random_gene() for _ in range(num_shapes)]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---------------------------
# Rendering
# ---------------------------
def render_genome(genome, size=DEFAULT_IMAGE_SIZE, mirror_prob=0.55, blur=0):
    """
    Render a genome to a PIL.Image (RGB).
    mirror_prob: probability to draw mirrored sibling for each shape (butterfly-like symmetry)
    """
    img = Image.new('RGBA', (size, size), (255, 255, 255, 255))
    for gene in genome:
        stype, x, y, sx, sy, angle, r, g, b, a = gene
        cx, cy = int(x * size), int(y * size)
        rx, ry = max(1, int(sx * size)), max(1, int(sy * size))

        layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(bbox, fill=(int(r), int(g), int(b), int(255 * a)))
        if abs(angle) > 0.001:
            layer = layer.rotate(angle, resample=Image.BICUBIC, center=(cx, cy), expand=False)
        img = Image.alpha_composite(img, layer)

        # mirrored sibling (for symmetry)
        if random.random() < mirror_prob:
            mx = 1.0 - x
            mcx, mcy = int(mx * size), cy
            mlayer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            mdraw = ImageDraw.Draw(mlayer)
            mbbox = [mcx - rx, mcy - ry, mcx + rx, mcy + ry]
            mr = clamp(int(r + np.random.normal(scale=6)), 0, 255)
            mg = clamp(int(g + np.random.normal(scale=6)), 0, 255)
            mb = clamp(int(b + np.random.normal(scale=6)), 0, 255)
            ma = clamp(a + np.random.normal(scale=0.03), 0.02, 0.98)
            mdraw.ellipse(mbbox, fill=(mr, mg, mb, int(255 * ma)))
            if abs(angle) > 0.001:
                mlayer = mlayer.rotate(-angle, resample=Image.BICUBIC, center=(mcx, mcy), expand=False)
            img = Image.alpha_composite(img, mlayer)

    if blur and blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    # paste onto white background and return RGB
    background = Image.new('RGB', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    return background

# ---------------------------
# Fitness functions (normalized & less noisy)
# ---------------------------
def _normalized_contrast(arr):
    """Normalize standard deviation of grayscale image to [0,1]."""
    std = arr.std()
    return float(clamp(std / 127.5, 0.0, 1.0))

def _normalized_symmetry(arr):
    """Compute left-right symmetry in [0,1] (1 is perfectly symmetric)."""
    h, w = arr.shape
    left = arr[:, :w//2]
    right = np.fliplr(arr[:, w - w//2:])
    minw = min(left.shape[1], right.shape[1])
    if minw <= 0:
        return 0.0
    left = left[:, :minw]
    right = right[:, :minw]
    diff = np.mean(np.abs(left - right))
    sym = 1.0 - (diff / 127.5)
    return float(clamp(sym, 0.0, 1.0))

def fitness_contrast_symmetry(pil_img, weight_symmetry=0.6, weight_contrast=0.4):
    """
    Normalized fitness in [0,1]. Blend of symmetry (left-right) and contrast.
    """
    img = pil_img.convert('L').resize((120, 120))
    arr = np.array(img).astype(np.float32)
    contrast_n = _normalized_contrast(arr)
    symmetry_n = _normalized_symmetry(arr)
    score = weight_symmetry * symmetry_n + weight_contrast * contrast_n
    return float(clamp(score, 0.0, 1.0))

def fitness_entropy(pil_img):
    img = pil_img.convert('L').resize((128, 128))
    hist = np.array(img.histogram(), dtype=float)
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(clamp(entropy / 8.0, 0.0, 1.0))

# ---------------------------
# Utility: smoothing time-series of fitnesses
# ---------------------------
def smooth_scores(scores, window=3):
    if window <= 1 or len(scores) == 0:
        return np.array(scores, dtype=float)
    kernel = np.ones(window) / float(window)
    sm = np.convolve(np.array(scores, dtype=float), kernel, mode='same')
    return sm

# ---------------------------
# GA operators (with exploration-friendly behavior)
# ---------------------------
def evaluate_population(population, renderer=render_genome, scorer=fitness_contrast_symmetry, size=DEFAULT_IMAGE_SIZE, smooth_window=0, inject_random_rate=0.0):
    """
    Render population and compute fitness scores.
    - smooth_window: moving-average smoothing of scores (for plotting)
    - inject_random_rate: fraction of population replaced with new random genomes (diversity)
    """
    pop = list(population)

    # optional injection of random genomes
    if inject_random_rate and inject_random_rate > 0.0:
        n_new = max(1, int(len(pop) * inject_random_rate))
        for _ in range(n_new):
            idx = random.randrange(len(pop))
            pop[idx] = random_genome(len(pop[idx])) if isinstance(pop[idx], list) else random_genome()

    images = []
    for g in pop:
        try:
            img = renderer(g, size=size)
        except TypeError:
            img = renderer(g)
        images.append(img)

    scores = np.array([scorer(im) for im in images], dtype=float)

    if smooth_window and smooth_window > 1:
        scores = smooth_scores(scores, window=smooth_window)

    return images, scores

def tournament_select(population, fitnesses, k=3):
    idx = np.random.randint(0, len(population), size=k)
    winner_idx = int(idx[np.argmax(fitnesses[idx])])
    return population[winner_idx]

def crossover(parent_a, parent_b, mix_prob=0.5):
    """
    Improved crossover:
    - Uniform gene-level selection (mix_prob)
    - For color fields, small cross-blends are possible
    """
    child = []
    for ga, gb in zip(parent_a, parent_b):
        gene = list(ga) if random.random() < mix_prob else list(gb)

        # occasionally blend colors instead of picking one parent exactly
        if random.random() < 0.15:
            # blend rgb channels
            r = int((ga[6] + gb[6]) / 2 + np.random.normal(scale=8))
            g = int((ga[7] + gb[7]) / 2 + np.random.normal(scale=8))
            b = int((ga[8] + gb[8]) / 2 + np.random.normal(scale=8))
            gene[6] = clamp(r, 0, 255)
            gene[7] = clamp(g, 0, 255)
            gene[8] = clamp(b, 0, 255)
        child.append(gene)
    return child

def mutate(genome, rate=0.12, scale=0.15, occasional_big_mutation_prob=0.08):
    """
    Mutate genome:
    - rate: per-gene mutation probability
    - scale: typical gaussian scale for continuous fields
    - occasional_big_mutation_prob: with this chance apply larger mutation to a gene
    """
    for gene in genome:
        if random.random() < rate:
            field = random.randrange(len(gene))
            big = (random.random() < occasional_big_mutation_prob)
            local_scale = scale * (3.0 if big else 1.0)

            if field == 0:
                gene[0] = random.choice([0, 1])
            elif field in [1, 2]:
                gene[field] = clamp(gene[field] + np.random.normal(scale=local_scale), 0.0, 1.0)
            elif field in [3, 4]:
                gene[field] = clamp(gene[field] + np.random.normal(scale=local_scale), 0.005, 0.6)
            elif field == 5:
                gene[field] = (gene[field] + np.random.normal(scale=local_scale * 180)) % 360
            elif field in [6, 7, 8]:
                gene[field] = int(clamp(int(gene[field] + np.random.normal(scale=local_scale * 255)), 0, 255))
            elif field == 9:
                gene[field] = clamp(gene[field] + np.random.normal(scale=local_scale), 0.02, 0.98)
    return genome

# ---------------------------
# High-level evolve helper
# ---------------------------
def evolve_population(population, selected_parents, pop_size, num_shapes,
                      elitism=2, mutation_rate=0.12, mutation_scale=0.15,
                      immigrant_rate=0.05):
    """
    Construct the next generation:
    - Keep `elitism` best individuals (by evaluating fitness once here)
    - Breed children by randomly choosing parents from selected_parents
    - Mutate children with given rate/scale (with occasional large jumps)
    - Replace a small fraction with random immigrants (immigrant_rate) to restore diversity
    Returns new_population (list of genomes).
    """
    # Evaluate current population to get elites
    _, fits = evaluate_population(population)
    new_pop = []

    # Elitism: keep top elites
    if elitism > 0:
        elite_idx = list(np.argsort(fits)[-elitism:])
        for ei in elite_idx:
            new_pop.append([list(g) for g in population[int(ei)]])

    # Fill remaining slots with offspring
    while len(new_pop) < pop_size:
        pa = random.choice(selected_parents)
        pb = random.choice(selected_parents)
        child = crossover(pa, pb)
        child = mutate(child, rate=mutation_rate, scale=mutation_scale)
        # ensure gene count
        if len(child) != num_shapes:
            if len(child) > num_shapes:
                child = child[:num_shapes]
            else:
                while len(child) < num_shapes:
                    child.append(random_gene())
        new_pop.append(child)

    # Immigrants: replace a few with brand-new random genomes
    if immigrant_rate and immigrant_rate > 0.0:
        n_new = max(1, int(pop_size * immigrant_rate))
        for _ in range(n_new):
            idx = random.randrange(len(new_pop))
            new_pop[idx] = random_genome(num_shapes)

    return new_pop

# ---------------------------
# Utilities for saving GIFs
# ---------------------------
def pil_images_to_gif(pil_images, out_path, duration=600):
    """Save a list of PIL images as a GIF."""
    if not pil_images:
        raise ValueError("No images passed to create GIF")
    frames = [im.convert("P", palette=Image.ADAPTIVE) for im in pil_images]
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    return out_path

def pil_image_to_bytes(pil_img, fmt="PNG"):
    """Convert PIL image to bytes (for Streamlit download)."""
    buff = io.BytesIO()
    pil_img.save(buff, format=fmt)
    buff.seek(0)
    return buff.getvalue()
