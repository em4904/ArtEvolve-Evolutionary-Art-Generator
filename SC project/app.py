# app.py — Interactive ArtEvolve with reactive controls + parent-history comparisons
import streamlit as st
import os
import random
from datetime import datetime
import numpy as np

from artevolve_core import (
    random_genome, render_genome, evaluate_population,
    crossover, mutate, pil_images_to_gif, pil_image_to_bytes,
    fitness_contrast_symmetry, create_population
)

st.set_page_config(page_title="ArtEvolve — Interactive GA", layout="wide")

# -------------------------
# Sidebar controls (all reactive)
# -------------------------
st.sidebar.title("ArtEvolve Controls")

pop_size = st.sidebar.slider("Population", min_value=6, max_value=24, value=12, step=2)
num_shapes = st.sidebar.slider("Shapes per artwork", min_value=4, max_value=30, value=12, step=1)
mutation_rate = st.sidebar.slider("Mutation rate", 0.0, 0.5, 0.18, 0.01)
mutation_scale = st.sidebar.slider("Mutation scale", 0.0, 1.0, 0.22, 0.01)
elitism = st.sidebar.slider("Elitism (keep top)", 0, 4, 2)
mirror_prob = st.sidebar.slider("Symmetry probability (mirror)", 0.0, 1.0, 0.55, 0.05)
image_size = st.sidebar.selectbox("Image size (px)", [200, 256, 300, 384, 512], index=2)

st.sidebar.markdown("---")
st.sidebar.write("Pick exactly 2 parents (or use Auto-select). Then click **Next Generation**.")
st.sidebar.write("Mutation & elitism affect the *next* generation; mirror/image size apply immediately to rendering.")

# -------------------------
# Session-state defaults and reactive param updates
# -------------------------
if "params" not in st.session_state:
    st.session_state.params = {
        "pop_size": pop_size,
        "num_shapes": num_shapes,
        "mutation_rate": mutation_rate,
        "mutation_scale": mutation_scale,
        "elitism": elitism,
        "mirror_prob": mirror_prob,
        "image_size": image_size
    }

# If the core structural params (pop_size or num_shapes) change, reinitialize population.
if (st.session_state.params["pop_size"] != pop_size) or (st.session_state.params["num_shapes"] != num_shapes):
    st.session_state.params.update({
        "pop_size": pop_size,
        "num_shapes": num_shapes,
        "mutation_rate": mutation_rate,
        "mutation_scale": mutation_scale,
        "elitism": elitism,
        "mirror_prob": mirror_prob,
        "image_size": image_size
    })
    # create new population with new sizes
    st.session_state.population = create_population(pop_size, num_shapes)
    st.session_state.generation = 0
    st.session_state.best_images = []
    st.session_state.best_history = []
    st.session_state.parent_history = []  # clear parent history
    st.session_state.pop_version = st.session_state.get("pop_version", 0) + 1

# If only non-structural params (mutation/mirror/image_size) changed, update params and
# bump pop_version when rendering parameters change (image_size or mirror_prob) so images refresh.
params_changed = False
for k, v in [("mutation_rate", mutation_rate), ("mutation_scale", mutation_scale),
             ("elitism", elitism), ("mirror_prob", mirror_prob), ("image_size", image_size)]:
    if st.session_state.params.get(k) != v:
        st.session_state.params[k] = v
        params_changed = True

if params_changed:
    # if rendering params changed, force re-evaluation so displayed images update
    if st.session_state.params["mirror_prob"] != mirror_prob or st.session_state.params["image_size"] != image_size:
        st.session_state.pop_version = st.session_state.get("pop_version", 0) + 1

# ensure population exists
if "population" not in st.session_state:
    st.session_state.population = create_population(pop_size, num_shapes)
    st.session_state.generation = 0
    st.session_state.best_images = []
    st.session_state.best_history = []
    st.session_state.parent_history = []
    st.session_state.pop_version = 0

# -------------------------
# Renderer for evaluate_population (must accept size kwarg)
# -------------------------
def renderer_fn(g, size=None):
    size = size or st.session_state.params["image_size"]
    return render_genome(g, size=size, mirror_prob=st.session_state.params["mirror_prob"])

# -------------------------
# Cached evaluation per population version
# -------------------------
if "last_evaluated_version" not in st.session_state:
    st.session_state.last_evaluated_version = -1

if st.session_state.last_evaluated_version != st.session_state.pop_version:
    imgs, fits = evaluate_population(st.session_state.population, renderer=renderer_fn, scorer=fitness_contrast_symmetry, size=st.session_state.params["image_size"])
    st.session_state.current_images = imgs
    st.session_state.current_fits = fits
    st.session_state.last_evaluated_version = st.session_state.pop_version
else:
    imgs = st.session_state.current_images
    fits = st.session_state.current_fits

# -------------------------
# Top actions
# -------------------------
col1, col2, col3, col4 = st.columns([1,1,1,2])
with col1:
    if st.button("Reset Evolution"):
        st.session_state.population = create_population(pop_size, num_shapes)
        st.session_state.generation = 0
        st.session_state.best_images = []
        st.session_state.best_history = []
        st.session_state.parent_history = []
        st.session_state.pop_version = st.session_state.get("pop_version", 0) + 1
        st.rerun()

with col2:
    if st.button("Auto-select top 2"):
        # find indices of top 2 fittest individuals
        top2_idx = list(np.argsort(fits)[-2:])
        st.session_state._auto_select_indices = top2_idx

        # mark their checkbox states explicitly so they stay checked
        gen = st.session_state.generation
        for i in top2_idx:
            key = f"pick_gen{gen}_idx{i}"
            st.session_state[key] = True

        st.success(f"Auto-selected top 2: {top2_idx}")
        st.rerun()


with col3:
    if st.button("Save best-so-far GIF"):
        if st.session_state.best_images:
            run_dir = os.path.join("artevolve_app_outputs", "runs")
            os.makedirs(run_dir, exist_ok=True)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(run_dir, f"evolution_{now}.gif")
            try:
                pil_images_to_gif(st.session_state.best_images, gif_path, duration=600)
                st.success(f"Saved GIF: {gif_path}")
            except Exception as e:
                st.error(f"GIF creation failed: {e}")
        else:
            st.warning("No images saved yet (run some generations).")

with col4:
    st.write(f"**Generation:** {st.session_state.generation}")

st.markdown("----")
st.title("🦋 ArtEvolve — Interactive Evolution (Select favorites)")

# -------------------------
# Display population + selection
# -------------------------
st.subheader("Population (pick exactly 2 parents)")
cols = st.columns(4)
checkbox_keys = []

# create generation-specific checkboxes so they reset only after evolving
for i, img in enumerate(imgs):
    c = cols[i % 4]
    with c:
        st.image(img, use_container_width=True)
        score_text = f" (score: {fits[i]:.2f})"
        key = f"pick_gen{st.session_state.generation}_idx{i}"
        default = False
        if "_auto_select_indices" in st.session_state and i in st.session_state._auto_select_indices:
            default = True
        picked = st.checkbox(f"Pick #{i}{score_text}", key=key, value=default)
        checkbox_keys.append((i, key))

# -------------------------
# Next Generation logic + Parent history tracking
# -------------------------
st.markdown("---")
next_col1, next_col2 = st.columns([1,1])

with next_col1:
    if st.button("Next Generation (use selected as parents)"):
        selected = [i for i, k in checkbox_keys if st.session_state.get(k, False)]
        if not selected and "_auto_select_indices" in st.session_state:
            selected = list(st.session_state._auto_select_indices)

        if len(selected) != 2:
            st.warning("Please select exactly 2 parents or use Auto-select top 2.")
        else:
            a_idx, b_idx = selected
            parent_a = st.session_state.population[a_idx]
            parent_b = st.session_state.population[b_idx]

            # Save parent images & fitness into history BEFORE evolving
            parent_entry = {
                "generation": st.session_state.generation,
                "parent_indices": [a_idx, b_idx],
                "parent_images": [imgs[a_idx], imgs[b_idx]],
                "parent_fitness": [float(fits[a_idx]), float(fits[b_idx])]
            }

            # build new population with elitism and breeding from selected parents
            new_pop = []
            if elitism > 0:
                elite_idx = list(np.argsort(fits)[-elitism:])
                for ei in elite_idx:
                    new_pop.append([list(g) for g in st.session_state.population[int(ei)]])  # copy elites

            while len(new_pop) < pop_size:
                pa = random.choice([parent_a, parent_b])
                pb = random.choice([parent_a, parent_b])
                child = crossover(pa, pb)
                child = mutate(child, rate=mutation_rate, scale=mutation_scale)
                # ensure gene count
                if len(child) != num_shapes:
                    if len(child) > num_shapes:
                        child = child[:num_shapes]
                    else:
                        while len(child) < num_shapes:
                            child.append(random_genome(1)[0])
                new_pop.append(child)

            # update population + metadata
            st.session_state.population = new_pop
            st.session_state.generation += 1
            st.session_state.pop_version = st.session_state.get("pop_version", 0) + 1

            # evaluate new population immediately and save best image
            new_imgs, new_fits = evaluate_population(st.session_state.population, renderer=renderer_fn, scorer=fitness_contrast_symmetry, size=st.session_state.params["image_size"])
            best_idx = int(np.argmax(new_fits))
            best_img = new_imgs[best_idx]

            # record best child in parent history entry (so comparison is Parent A | Parent B -> Best Child)
            parent_entry["best_child"] = best_img
            parent_entry["best_child_fitness"] = float(new_fits[best_idx])

            # append to history and best_images/history for export
            st.session_state.parent_history.append(parent_entry)
            st.session_state.best_images.append(best_img)
            st.session_state.best_history.append(float(new_fits[best_idx]))

            # clear auto-select marker
            st.session_state.pop("_auto_select_indices", None)

            # force re-evaluation next render and refresh UI
            st.rerun()

with next_col2:
    if st.button("Randomize population"):
        st.session_state.population = create_population(pop_size, num_shapes)
        st.session_state.generation = 0
        st.session_state.best_images = []
        st.session_state.best_history = []
        st.session_state.parent_history = []
        st.session_state.pop_version = st.session_state.get("pop_version", 0) + 1
        st.rerun()

# -------------------------
# Parent history & comparisons (Option A layout)
# -------------------------
st.markdown("---")
st.subheader("Parent History & Comparisons")
if st.session_state.parent_history:
    # Show most recent at top
    for entry in reversed(st.session_state.parent_history):
        gen = entry["generation"]
        with st.expander(f"Generation {gen} — Parents {entry['parent_indices']} (click to expand)"):
            pa_img, pb_img = entry["parent_images"]
            pfa, pfb = entry["parent_fitness"]
            bc_img = entry.get("best_child")
            bcf = entry.get("best_child_fitness", None)

            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.image(pa_img, use_container_width=True)
                st.caption(f"Parent A — idx {entry['parent_indices'][0]} — fitness {pfa:.3f}")
            with c2:
                st.image(pb_img, use_container_width=True)
                st.caption(f"Parent B — idx {entry['parent_indices'][1]} — fitness {pfb:.3f}")
            with c3:
                if bc_img is not None:
                    st.image(bc_img, use_container_width=True)
                    st.caption(f"Best child (from this breeding) — fitness {bcf:.3f}")
                else:
                    st.write("No child recorded.")
else:
    st.info("No parent history yet. Select parents and evolve to populate history.")

# -------------------------
# Best-of-generation history + export
# -------------------------
st.markdown("---")
st.subheader("Best-of-generation history")
if st.session_state.best_images:
    last_best = st.session_state.best_images[-1]
    st.image(last_best, caption=f"Best (Gen {st.session_state.generation - 1})", use_container_width=False)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(st.session_state.best_history, marker="o")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (automated)")
        st.pyplot(fig)
    except Exception:
        pass
else:
    st.info("No best history yet. Press 'Next Generation' after selecting favorites to save best images.")

st.markdown("---")
st.subheader("Export")
if st.button("Download best GIF (current best_images)"):
    if st.session_state.best_images:
        run_dir = os.path.join("artevolve_app_outputs", "runs")
        os.makedirs(run_dir, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(run_dir, f"evolution_{now}.gif")
        try:
            pil_images_to_gif(st.session_state.best_images, gif_path, duration=600)
            with open(gif_path, "rb") as f:
                st.download_button("Download GIF", data=f, file_name=f"evolution_{now}.gif", mime="image/gif")
        except Exception as e:
            st.error(f"Could not create GIF: {e}")
    else:
        st.warning("No images to create GIF.")

if st.session_state.best_images:
    last_buf = pil_image_to_bytes(st.session_state.best_images[-1], fmt="PNG")
    st.download_button("Download latest best (PNG)", data=last_buf, file_name=f"best_gen_{st.session_state.generation - 1}.png", mime="image/png")

st.markdown("----")
st.caption("Tip: pick exactly 2 favorite artworks per generation; press Next Generation to breed. Parent history appears below as comparisons (Parent A | Parent B → Best Child).")
