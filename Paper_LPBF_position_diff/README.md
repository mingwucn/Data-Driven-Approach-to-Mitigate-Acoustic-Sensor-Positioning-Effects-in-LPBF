## ROAD map
This is the specific, execution-level roadmap for the **Data Analysis Stage**.

Since you have **Healthy Data** (Config 1 & 2) and **Defect Data** (Pore vs. No Pore), your goal in this stage is to mathematically prove that **Position = Filter** and **Defect = Source**, and that they are separable.

### **Phase 1: Quantifying the "Observer Effect" (The Problem)**
**Objective:** Proving that the "Context Shift" is a structural artifact and not random noise.
**Input:** Healthy Data (Config 1: 5 positions, Config 2: 5 positions).

* **Step 1.1: Spectral Normalization**
    * **Action:** Convert all raw acoustic signals to the Frequency Domain (FFT).
    * **Critical Detail:** Use a **Log-Mel Spectrogram** or **Power Spectral Density (PSD)** scale. Linear scales often hide the "floor" noise where artifacts hide. Ensure both configurations are normalized to their own local maxima (0 to 1 scaling) to compare shapes, not absolute voltages.
* **Step 1.2: The "Variance Fingerprint"**
    * **Action:** For Config 1, calculate the **Spectral Variance vector** ($\sigma^2_f$) across the 5 positions. Do the same for Config 2.
    * **Mathematical Check:**
        $$\sigma^2(f) = \frac{1}{N} \sum_{pos=1}^{5} (PSD_{pos}(f) - \mu(f))^2$$
    * **The Comparison:** Plot $\sigma^2_{config1}(f)$ and $\sigma^2_{config2}(f)$ on the same graph.
    * **Expected Outcome:** You will see distinct peaks in variance. Crucially, the **peaks will be at different frequencies** for Config 1 vs. Config 2.
    * **Scientific Claim:** "The 'Red Zones' (high variance) shift when the hardware changes, proving they are **structural eigenmodes** ($H$), not process universals ($S$)."

### **Phase 2: The "Tautology" Check (The Mechanism)**
**Objective:** Proving the AI is a "Variance Detector."
**Input:** Healthy Data + Trained CNN Models for Position Classification.

* **Step 2.1: Grad-CAM Extraction**
    * [cite_start]**Action:** Train your 1D-CNN (as described in your manuscript [cite: 18, 96]) to classify the 5 positions. [cite_start]Run **Grad-CAM** [cite: 154] on the test set to get the "Relevance Scores" for each frequency bin.
* **Step 2.2: The "Correlation" Plot (Figure 3 in new plan)**
    * **Action:** Normalize the Grad-CAM weights (0-1). Normalize the Statistical Variance from Step 1.2 (0-1).
    * **Analysis:** Calculate the **Pearson Correlation Coefficient** ($r$) between the AI attention and the Statistical Variance.
    * **Expected Outcome:** You want $r > 0.8$.
    * **Scientific Claim:** "The Neural Network converges on the statistically most volatile frequency bands. It effectively 'learns' the baseplate's resonance pattern to identify position." (This defends against the 'Black Box' critique).

### **Phase 3: Constructing the "Trust Map" (The Solution)**
**Objective:** Creating the filter that separates "Context" from "Content."
**Input:** Variance vectors from Phase 1.

* **Step 3.1: Defining the Mask**
    * **Action:** Define a threshold (e.g., frequencies with variance $>$ 75th percentile).
    * **Definition:**
        * **Red Zone (Masked):** Frequency $f$ where $\sigma^2(f) > Threshold$.
        * **Green Zone (Safe):** Frequency $f$ where $\sigma^2(f) < Threshold$.
* **Step 3.2: The "Collapse" Test**
    * **Action:** Apply the inverse mask (keep only Green Zones) to your raw data from all 5 positions.
    * **Analysis:** Calculate the **Cosine Similarity** between the spectra of Position 1 and Position 5 *after* masking.
    * **Expected Outcome:** The similarity should jump from low (e.g., 0.6) to high (e.g., 0.95).
    * **Scientific Claim:** "By masking the geometric artifacts, we collapse the position-dependent manifolds into a single, invariant process baseline."

### **Phase 4: The "Pore Preservation" Validation (The Proof)**
**Objective:** Proving you didn't delete the defect signal.
**Input:** Defect Data (Pore vs. No Pore).

* **Step 4.1: Isolating the Defect Signature**
    * **Action:** Take your Defect Dataset. Compute the **Mean Spectral Difference**:
        $$\Delta_{defect}(f) = | \text{Mean}(PSD_{pore}) - \text{Mean}(PSD_{healthy}) |$$
    * **Visual:** This curve ($\Delta_{defect}$) represents the acoustic sound of a pore.
* **Step 4.2: The "Overlay" (Figure 4 in new plan)**
    * **Action:** Plot the **Trust Map** (Red Zones from Step 3.1) and the **Defect Signature** (Blue Curve from Step 4.1) on the same X-axis.
    * **Analysis:** Measure the **Spectral Intersection**.
        * *Scenario A:* The Defect peak is at 80 kHz. The Red Zone is at 150 kHz. -> **SUCCESS.**
        * *Scenario B:* The Defect peak is at 150 kHz. The Red Zone is at 150 kHz. -> **BLIND SPOT.**
    * **Expected Outcome:** You likely have a broadband defect signal (pores usually create broadband noise or specific high-frequency signatures). You will show that *enough* of the defect signal falls into the "Green Zones" to remain detectable.
    * **Scientific Claim:** "Disentanglement achieved. While geometric artifacts dominate specific narrow bands, the broadband nature of defect dynamics ensures detection viability within the 'Safe Zones'."

### **Summary of Outcomes for the Paper**

| Analysis Step | Visualization (Figure) | Scientific Conclusion |
| :--- | :--- | :--- |
| **1. Variance Comparison** | Heatmap of Config 1 vs Config 2 | "The artifacts are structural, not distinct to the process." |
| **2. AI Correlation** | Overlay of Grad-CAM & Variance | "The AI learns to identify the structure, not the physics." |
| **3. Trust Map** | "Before & After" Spectral Collapse | "We can mathematically remove the structure." |
| **4. Defect Overlay** | Red (Artifact) vs Blue (Defect) curves | "The defect signal survives the cleaning process." |

**Would you like me to start by generating the Python code for "Phase 1: Variance Comparison" so you can immediately plug in your CSV/data files?**