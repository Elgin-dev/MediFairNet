# 🏆 1st Place Winner — Best Overall Project @ ML Empowerment Build Challenge 2026

# 🏥 MediFairNet

> **Unbiased, Trust-Calibrated Multi-Modal Clinical Diagnostics** > Powered by PyTorch, Accelerated via NVIDIA CUDA, Integrated with BioBERT.

---

## 🛠️ Technology Stack & Tools

<table>
  <tr>
    <td bgcolor="#0A192F" align="center" style="padding: 15px;">
      <font color="#64FFDA" size="4"><b>🧠 Core AI Frameworks</b></font><br>
      <code>python</code> &nbsp;|&nbsp; <code>pytorch</code> &nbsp;|&nbsp; <code>torchvision</code> &nbsp;|&nbsp; <code>huggingface-transformers</code>
    </td>
  </tr>
  <tr>
    <td bgcolor="#172A45" align="center" style="padding: 15px;">
      <font color="#F778BA" size="4"><b>🏎️ Compute Acceleration & Logging</b></font><br>
      <code>nvidia-cuda</code> &nbsp;|&nbsp; <code>tensorboard</code>
    </td>
  </tr>
  <tr>
    <td bgcolor="#0A192F" align="center" style="padding: 15px;">
      <font color="#38BDF8" size="4"><b>🌐 Backend Architecture</b></font><br>
      <code>fastapi</code> &nbsp;|&nbsp; <code>uvicorn</code> &nbsp;|&nbsp; <code>pillow</code>
    </td>
  </tr>
  <tr>
    <td bgcolor="#172A45" align="center" style="padding: 15px;">
      <font color="#A7F3D0" size="4"><b>📊 Data Engineering</b></font><br>
      <code>pandas</code> &nbsp;|&nbsp; <code>numpy</code>
    </td>
  </tr>
</table>

---

## 💡 The Core Innovation

MediFairNet addresses a systemic flaw in medical artificial intelligence: **Demographic Shortcutting**. Standard deep neural networks train on medical images but inadvertently pick up on non-pathological features (e.g., patient sex, age, scanner positioning artifacts) to make rapid diagnostic guesses. 

MediFairNet completely re-engineers this pipeline. It actively scrubs demographic bias out at the latent vector layer using **Adversarial Minimax Games**, combines clean visual data with **BioBERT** text embeddings, and tracks its own **Epistemic Uncertainty** using Monte Carlo Dropout to ensure a model never masks clinical confusion with an overconfident false guess.

---

## 🗺️ System Architecture

Our end-to-end processing pipeline runs through a four-stage neural stack:
