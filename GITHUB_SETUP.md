# 🚀 Panduan Lengkap Setup GitHub untuk Advanced Shape Detection

## 📋 Langkah-langkah Setup GitHub

### 1. **Persiapkan Akun GitHub**
- Pastikan Anda sudah memiliki akun GitHub
- Jika belum, buat akun di [github.com](https://github.com)

### 2. **Konfigurasi Git (Sudah Selesai)**
```bash
# Git sudah dikonfigurasi dengan:
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. **Setup GitHub Authentication**

#### Opsi A: Personal Access Token (Recommended)
1. Buka [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. Klik "Generate new token (classic)"
3. Beri nama token (misal: "Advanced Shape Detection")
4. Pilih scopes:
   - ✅ `repo` (Full control of private repositories)
   - ✅ `workflow` (Update GitHub Action workflows)
5. Klik "Generate token"
6. **COPY TOKEN** (jangan lupa, hanya muncul sekali!)

#### Opsi B: GitHub CLI
```bash
# Install GitHub CLI
winget install GitHub.cli

# Login dengan GitHub CLI
gh auth login
```

### 4. **Buat Repository di GitHub**

#### Cara Manual:
1. Buka [github.com](https://github.com)
2. Klik tombol "+" di pojok kanan atas
3. Pilih "New repository"
4. Isi:
   - **Repository name**: `advanced-shape-detection`
   - **Description**: `Advanced Shape Detection and Coloring System for Computer Vision`
   - **Visibility**: Public (atau Private sesuai preferensi)
   - ✅ Add a README file
   - ✅ Add .gitignore (Python)
   - ✅ Choose a license (MIT License)
5. Klik "Create repository"

#### Cara dengan GitHub CLI:
```bash
gh repo create advanced-shape-detection --public --description "Advanced Shape Detection and Coloring System for Computer Vision" --add-readme --gitignore Python --license MIT
```

### 5. **Push Repository ke GitHub**

#### Jika repository sudah ada di GitHub:
```bash
# Tambahkan remote origin
git remote add origin https://github.com/YOUR_USERNAME/advanced-shape-detection.git

# Push ke GitHub
git branch -M main
git push -u origin main
```

#### Jika repository kosong di GitHub:
```bash
# Tambahkan remote origin
git remote add origin https://github.com/YOUR_USERNAME/advanced-shape-detection.git

# Push semua branch
git push -u origin --all

# Push semua tags
git push -u origin --tags
```

### 6. **Setup GitHub Actions**

Setelah push, GitHub Actions akan otomatis berjalan karena file `.github/workflows/ci.yml` sudah ada.

### 7. **Verifikasi Setup**

1. Buka repository di GitHub
2. Cek tab "Actions" untuk melihat CI/CD pipeline
3. Cek tab "Issues" untuk melihat template
4. Cek tab "Pull requests" untuk workflow

## 🔧 Troubleshooting

### Error: Authentication failed
```bash
# Gunakan Personal Access Token sebagai password
git push origin main
# Username: YOUR_GITHUB_USERNAME
# Password: YOUR_PERSONAL_ACCESS_TOKEN
```

### Error: Repository not found
```bash
# Pastikan URL repository benar
git remote set-url origin https://github.com/YOUR_USERNAME/advanced-shape-detection.git
```

### Error: Permission denied
```bash
# Pastikan token memiliki permission yang cukup
# Atau gunakan SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"
```

## 📝 Langkah Selanjutnya

### 1. **Update README.md**
- Ganti `yourusername` dengan username GitHub Anda
- Update link repository

### 2. **Setup Branch Protection**
1. Buka repository settings
2. Branches > Add rule
3. Branch name pattern: `main`
4. ✅ Require pull request reviews
5. ✅ Require status checks to pass

### 3. **Setup Issue Templates**
- Template sudah tersedia di `.github/ISSUE_TEMPLATE/`

### 4. **Setup Release**
```bash
# Buat release v1.0.0
gh release create v1.0.0 --title "Initial Release" --notes "First release of Advanced Shape Detection System"
```

## 🎯 Fitur yang Sudah Siap

✅ **CI/CD Pipeline** dengan GitHub Actions  
✅ **Code Quality** dengan flake8 dan pytest  
✅ **Docker Support** dengan Dockerfile dan docker-compose  
✅ **Documentation** lengkap dengan README, CONTRIBUTING, dll  
✅ **Security** dengan SECURITY.md dan CODE_OF_CONDUCT  
✅ **Testing** dengan pytest dan coverage  
✅ **Packaging** dengan setup.py dan pyproject.toml  

## 📞 Bantuan

Jika mengalami masalah:
1. Cek [GitHub Documentation](https://docs.github.com/)
2. Buka issue di repository
3. Hubungi maintainer

---

**Selamat! Repository Anda siap untuk collaboration dan deployment! 🎉**
