# Panduan & Pengingat Pengembangan Agen (AGENTS.md)

Dokumen ini berisi pedoman operasional, batasan teknis, dan arsitektur kritis untuk pengembangan dan pemeliharaan repository **SayMaven/Fooocus**.

---

## 1. Batasan Lingkungan & Hardware (Hardware Constraints)

- **Komputer Lokal (Windows User):**
  - Spesifikasi rendah (RAM 8 GB, VRAM 4 GB, tanpa CUDA PyTorch penuh).
  - **DILARANG** menjalankan Fooocus, proses inferensi AI, atau pemuatan checkpoint model secara lokal.
  - Perangkat lokal hanya digunakan untuk: *editing file*, analisis kode (*static analysis*), pengujian sintaks cepat (`py_compile`), dan operasi Git (`commit`, `push`).
- **Google Colab (Remote Target):**
  - Lingkungan eksekusi utama (GPU NVIDIA Tesla T4 16 GB, RAM ~13 GB).
  - Semua inferensi model, pembuatan gambar, dan verifikasi runtime dilakukan di Colab.

---

## 2. Strategi Branch Git

- **Branch `main`:**
  - Versi stabil dasar. **JANGAN melakukan direct commit atau eksperimen langsung ke branch ini.**
- **Branch `colab-support`:**
  - Branch kerja stabil untuk adaptasi runtime Colab modern (Python 3.13+, CUDA 12.8+, Transformers 5.x).
- **Branch `anima-support`:**
  - Branch pengembangan aktif untuk integrasi arsitektur ganda: **SDXL (UNet)** dan **Anima (DiT)**.
  - Semua perubahan kode fitur Anima di-commit dan di-push ke branch ini.
  - Perubahan di branch ini TIDAK BOLEH merusak fungsionalitas SDXL atau perbaikan runtime Colab yang sudah ada di `colab-support`.
  - Folder referensi lokal `F:\CODE\forking\FooocusAnima` dijaga **100% READ ONLY**.

---

## 3. Catatan Teknis & Gotchas Runtime Modern Google Colab

### A. Kompatibilitas NumPy (`numpy>=1.26.0`)
- **Masalah Legacy:** Fooocus aslinya mengunci `numpy==1.26.4`. Di Python 3.13 Colab, memaksa downgrade ke `numpy<2.0.0` memicu pip mengunduh `.tar.gz` dan mengompilasi dari source (Meson/Ninja) yang memakan waktu 15–25 menit hingga terlihat *stuck*.
- **Kebijakan Saat Ini:** Di branch `colab-support` dan `anima-support`, syarat telah dilonggarkan ke `numpy>=1.26.0` (mendukung NumPy 2.x bawaan Colab).
- **Aturan:** **JANGAN memaksa downgrade ke `numpy<2.0.0` atau `numpy==1.26.4`**. Biarkan Fooocus memakai NumPy bawaan Colab secara langsung tanpa instalasi ulang.

### B. Auto-Update Git Tanpa `pygit2`
- `entry_with_update.py` telah diperbarui menggunakan Git CLI bawaan sistem (`git pull --ff-only`).
- Jangan kembalikan dependensi hard ke `pygit2` karena membutuhkan kompilasi binary `libgit2` yang sering gagal di Python 3.13.

### C. Bypass & Selektif Pip (`--skip-pip`)
- Argumen `--skip-pip` (dan environment variable `SKIP_PIP=1`) tersedia pada `launch.py` dan `args_manager.py`.
- Gunakan fitur ini di Colab jika dependensi sudah terpasang agar Fooocus tidak memicu loop verifikasi pip yang memakan waktu.

### D. Kompatibilitas `transformers` 5.x (Arsitektur CLIP yang Diratakan / Flattened)
- **Masalah:** Pada `transformers` versi 5.x, kelas `CLIPTextModel` tidak lagi memiliki submodule bertingkat `self.text_model`. Layer `embeddings`, `encoder`, dan `final_layer_norm` berada langsung di bawah `transformer`.
- **Akibat jika diabaikan:** Seluruh bobot CLIP text encoder tidak akan termuat saat checkpoint di-load (`extra` / `left over keys`), menyebabkan output gambar menjadi pola noise oranye. Selain itu, LoRA text encoder akan gagal dicocokkan.
- **Solusi yang harus dijaga:**
  1. `ldm_patched/modules/sd.py` (`load_model_weights`): Otomatis memetakan kunci `transformer.text_model.` ke `transformer.` saat arsitektur model terdeteksi *flattened*.
  2. `ldm_patched/modules/lora.py` (`model_lora_keys_clip`): Mendukung pencocokan kunci layer dengan maupun tanpa `text_model`.
  3. `modules/patch_clip.py`: Mempertahankan fallback `property(lambda self: self)` untuk akses atribut Python legacy (`.text_model.`).

---

## 4. Arsitektur Ganda: SDXL (UNet) + Anima (DiT)

Fooocus mendukung eksekusi dua arsitektur berbeda secara bersamaan melalui *dynamic model dispatch*:

### A. Perbedaan Inti Pipeline
1. **SDXL (UNet):**
   - Menggunakan dual CLIP (OpenCLIP + CLIP-L) yang termuat di checkpoint.
   - Latent 4-channel (`(B, 4, H/8, W/8)`).
   - VAE SDXL standar (4-channel).
   - Sampler: Karras, Euler, DPM++ 2M, CFG 4.0 - 7.0.
   - Fooocus V2 expansion aktif.
2. **Anima (DiT):**
   - Arsitektur backbone: DiT 28-block (`MiniTrainDIT` / `CosmosTransformer`).
   - Text Conditioning: Qwen3-0.6B (`models/clip/qwen_3_06b_base.safetensors`) + T5 Tokenizer (IDs target-side) + `LLMAdapter` 6-block bridge (`crossattn_emb` (B, 512, 1024)).
   - Latent format: Wan21 16-channel (`(B, 16, 1, H/8, W/8)`), normalisasi mean/std.
   - VAE: Qwen-Image VAE (`WanVAE`, 16-channel, `models/vae/qwen_image_vae.safetensors`).
   - Sampler: Flow Matching (Rectified Flow, `multiplier=1.0, shift=3.0`), `euler_ancestral` + `simple` scheduler, CFG 3.5 - 4.5.
   - Prompting: Danbooru tags + natural language, padding wajib ke 512 tokens (attention sink).
   - Fooocus V2 expansion dinonaktifkan otomatis untuk Anima (tidak cocok dengan token Danbooru).
   - Previewer 4-channel di-bypass (karena 16-channel akan menyebabkan crash tensor mismatch).

### B. Sampler Fallback & Reference Bootstrap ComfyUI
- Di `modules/core.py`, saat sampling model Anima dilakukan, jika menggunakan anisotropic filter Fooocus native dapat memicu crash pada 5D tensor (`torch.pad` reflect mode NotImplemented).
- Solusi: `_can_use_anima_reference_sampler()` secara otomatis mendeteksi model Anima dan melakukan bootstrap shallow reference ComfyUI sampler (`comfy.sample.sample`) bila tersedia.
- Model non-Anima (SDXL) langsung melewati guard ini dan tetap menggunakan Fooocus native sampler biasa.

---

## 5. Workflow Perubahan Kode & Pengujian

1. Lakukan modifikasi kode di branch `anima-support`.
2. Validasi sintaks Python:
   ```powershell
   python -m py_compile <file_yang_diubah>
   ```
3. Jalankan pengujian unit test preset:
   ```powershell
   python -m unittest tests/test_anima_preset.py
   ```
4. Commit dan push ke GitHub:
   ```powershell
   git commit -am "Deskripsi perubahan"
   git push origin anima-support
   ```
5. Instruksikan pengguna untuk me-restart sel Fooocus di Google Colab guna menguji fungsionalitasnya.
