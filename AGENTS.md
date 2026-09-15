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
  - Branch kerja aktif untuk adaptasi runtime Colab modern (Python 3.13+, CUDA 12.8+, Transformers 5.x).
  - Semua perubahan kode harus di-commit dan di-push ke `origin colab-support`.
  - Colab akan otomatis mendownload pembaruan via `git pull --ff-only` di `entry_with_update.py`.

---

## 3. Catatan Teknis & Gotchas Runtime Modern Google Colab

### A. Kompatibilitas NumPy (`numpy>=1.26.0`)
- **Masalah Legacy:** Fooocus aslinya mengunci `numpy==1.26.4`. Di Python 3.13 Colab, memaksa downgrade ke `numpy<2.0.0` memicu pip mengunduh `.tar.gz` dan mengompilasi dari source (Meson/Ninja) yang memakan waktu 15–25 menit hingga terlihat *stuck*.
- **Kebijakan Saat Ini:** Di branch `colab-support`, syarat telah dilonggarkan ke `numpy>=1.26.0` (mendukung NumPy 2.x bawaan Colab).
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

## 4. Workflow Perubahan Kode & Pengujian

1. Lakukan modifikasi kode di branch `colab-support`.
2. Validasi sintaks Python:
   ```powershell
   python -m py_compile <file_yang_diubah>
   ```
3. Commit dan push ke GitHub:
   ```powershell
   git commit -am "Deskripsi perubahan"
   git push origin colab-support
   ```
4. Instruksikan pengguna untuk me-restart sel Fooocus di Google Colab guna menguji fungsionalitasnya.
