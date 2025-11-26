
# 🎨✨ ArtEvolve – Evolutionary Art Generator  
### 🧬 Soft Computing Project • 🎛️ Interactive Streamlit App • 🎨 Real-Time Art Evolution

ArtEvolve is an **evolutionary art generator** where unique abstract artworks evolve over generations using **Genetic Algorithms (GAs)**.  
Users guide evolution by selecting parents, tuning parameters, and watching the art evolve live.

---

## 🚀 Features

- 🎛️ **Interactive Streamlit UI**  
- 🎨 **Procedurally generated artwork** (no external assets)  
- 🧬 **Genome-based rendering** (shapes, colors, transparency, symmetry)  
- 🔁 **GA Operators:** elitism, crossover, adaptive mutation  
- 🧠 **Fitness scoring:** contrast + left-right symmetry  
- 👨‍👩‍👧 **Pick exactly 2 parents** or auto-select best ones  
- 🧪 Adjustable parameters:
  - Population size  
  - Number of shapes  
  - Mutation rate & scale  
  - Symmetry probability  
  - Image size  
  - Elitism  
- 🖼️ **Parent history** (Parent A | Parent B → Best Child)  
- 🎥 **GIF export** of all best artworks  
- 📥 **Download last best image (PNG)**  

---

## 🧬 Genome Structure

Each artwork is a genome of shape-genes:

```

[type, x, y, sx, sy, angle, r, g, b, alpha]

```

Rendering includes:  
✨ ellipses • 🎨 rotation • 🫧 transparency • 🌈 color blending • 🦋 symmetry mirroring

---

## 🧠 Genetic Algorithm Workflow

### 🔹 1. Initialize Population  
Random genomes generated with `create_population()`.

### 🔹 2. Render & Score  
Each artwork is rendered and scored based on:  
- 🪞 **Symmetry**  
- 🎚️ **Contrast**

### 🔹 3. Select Parents  
Pick 2 manually or auto-select top 2.

### 🔹 4. Crossover  
Gene-level mixing + color blending.

### 🔹 5. Mutation  
Gaussian noise + occasional big mutations 🎇

### 🔹 6. Elitism  
Top individuals carried untouched to next generation.

### 🔹 7. Diversity Injection  
Random immigrants avoid stagnation.

---

## 📊 Fitness Formula

Final fitness score:

```

fitness = 0.6 * symmetry + 0.4 * contrast

````

---

## 📦 Installation

### 1️⃣ Create a virtual environment (recommended)

```bash
python -m venv artevolve_env
````

Activate:

```bash
# Windows
artevolve_env\Scripts\activate

# Mac/Linux
source artevolve_env/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

This opens the **ArtEvolve Dashboard** 🌈 in your browser.

---

## 🖥️ How to Use

* Adjust parameters in the sidebar
* Review current population 🎨
* Select **exactly 2 parents**
* Click **Next Generation**
* Watch the new generation evolve
* View **best-of-generation chart**
* Export a GIF or download the latest best PNG

---

## 🛠️ Tech Stack

* 🐍 Python
* 🌐 Streamlit
* 🖼️ Pillow (PIL)
* 🔢 NumPy
* 📈 Matplotlib
* 🤖 Genetic Algorithms
* 🧠 Soft Computing Concepts

---

## 🔮 Future Enhancements

* 🌈 Multi-objective aesthetics (entropy, harmony, minimalism)
* 🧠 Neural aesthetic scoring
* 🖌️ Custom brush / shape editor
* 📦 JSON genome export
* 📸 4K ultra-resolution render mode

---

## 📜 License

MIT License.

---

## 🙌 Credits

Developed by **Esha** 💛
Core implementation in:

* `app.py`
* `artevolve_core.py`
* `requirements.txt`


Just tell me!
```
