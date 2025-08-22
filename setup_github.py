#!/usr/bin/env python3
"""
GitHub Setup Script untuk Advanced Shape Detection Project
Otomatisasi setup repository GitHub dengan semua konfigurasi yang diperlukan.
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path


def run_command(command, check=True, capture_output=True):
    """Jalankan command dan return result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error menjalankan command: {command}")
        print(f"Error: {e}")
        return None


def check_git_config():
    """Cek konfigurasi Git."""
    print("🔍 Mengecek konfigurasi Git...")
    
    name = run_command("git config --global user.name")
    email = run_command("git config --global user.email")
    
    if name and email:
        print(f"✅ Git config: {name.stdout.strip()} <{email.stdout.strip()}>")
        return True
    else:
        print("❌ Git config belum diset")
        return False


def setup_git_config():
    """Setup konfigurasi Git."""
    print("⚙️ Setup konfigurasi Git...")
    
    name = input("Masukkan nama Anda: ").strip()
    email = input("Masukkan email GitHub Anda: ").strip()
    
    if name and email:
        run_command(f'git config --global user.name "{name}"')
        run_command(f'git config --global user.email "{email}"')
        print("✅ Git config berhasil diset")
        return True
    else:
        print("❌ Nama dan email tidak boleh kosong")
        return False


def check_github_cli():
    """Cek apakah GitHub CLI terinstall."""
    print("🔍 Mengecek GitHub CLI...")
    
    result = run_command("gh --version")
    if result and result.returncode == 0:
        print("✅ GitHub CLI terinstall")
        return True
    else:
        print("❌ GitHub CLI belum terinstall")
        return False


def install_github_cli():
    """Install GitHub CLI."""
    print("📦 Installing GitHub CLI...")
    
    if sys.platform == "win32":
        result = run_command("winget install GitHub.cli")
    elif sys.platform == "darwin":
        result = run_command("brew install gh")
    else:
        result = run_command("curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg")
        if result:
            result = run_command("echo 'deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main' | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null")
        if result:
            result = run_command("sudo apt update && sudo apt install gh")
    
    if result and result.returncode == 0:
        print("✅ GitHub CLI berhasil diinstall")
        return True
    else:
        print("❌ Gagal install GitHub CLI")
        return False


def login_github():
    """Login ke GitHub."""
    print("🔐 Login ke GitHub...")
    
    result = run_command("gh auth login --web")
    if result and result.returncode == 0:
        print("✅ Login GitHub berhasil")
        return True
    else:
        print("❌ Gagal login ke GitHub")
        return False


def create_repository():
    """Buat repository di GitHub."""
    print("📁 Membuat repository di GitHub...")
    
    repo_name = "advanced-shape-detection"
    description = "Advanced Shape Detection and Coloring System for Computer Vision"
    
    # Cek apakah repository sudah ada
    result = run_command(f"gh repo view {repo_name}")
    if result and result.returncode == 0:
        print(f"✅ Repository {repo_name} sudah ada")
        return True
    
    # Buat repository baru
    visibility = input("Pilih visibility (public/private) [public]: ").strip().lower()
    if not visibility:
        visibility = "public"
    
    cmd = f'gh repo create {repo_name} --{visibility} --description "{description}" --source . --remote origin --push'
    result = run_command(cmd)
    
    if result and result.returncode == 0:
        print(f"✅ Repository {repo_name} berhasil dibuat")
        return True
    else:
        print("❌ Gagal membuat repository")
        return False


def setup_remote():
    """Setup remote origin."""
    print("🔗 Setup remote origin...")
    
    # Cek apakah remote sudah ada
    result = run_command("git remote -v")
    if result and "origin" in result.stdout:
        print("✅ Remote origin sudah ada")
        return True
    
    username = input("Masukkan username GitHub Anda: ").strip()
    if not username:
        print("❌ Username tidak boleh kosong")
        return False
    
    repo_url = f"https://github.com/{username}/advanced-shape-detection.git"
    result = run_command(f"git remote add origin {repo_url}")
    
    if result:
        print("✅ Remote origin berhasil diset")
        return True
    else:
        print("❌ Gagal setup remote origin")
        return False


def push_to_github():
    """Push code ke GitHub."""
    print("🚀 Push code ke GitHub...")
    
    # Rename branch ke main
    run_command("git branch -M main")
    
    # Push ke GitHub
    result = run_command("git push -u origin main")
    
    if result and result.returncode == 0:
        print("✅ Code berhasil di-push ke GitHub")
        return True
    else:
        print("❌ Gagal push ke GitHub")
        return False


def setup_branch_protection():
    """Setup branch protection."""
    print("🛡️ Setup branch protection...")
    
    repo_name = "advanced-shape-detection"
    
    # Setup branch protection rules
    cmd = f'''gh api repos/:owner/{repo_name}/branches/main/protection \
        --method PUT \
        --field required_status_checks='{{"strict":true,"contexts":[]}}' \
        --field enforce_admins=true \
        --field required_pull_request_reviews='{{"required_approving_review_count":1}}' \
        --field restrictions=null'''
    
    result = run_command(cmd)
    
    if result and result.returncode == 0:
        print("✅ Branch protection berhasil diset")
        return True
    else:
        print("⚠️ Gagal setup branch protection (bisa diset manual)")
        return False


def create_initial_release():
    """Buat initial release."""
    print("🏷️ Membuat initial release...")
    
    cmd = '''gh release create v1.0.0 \
        --title "Initial Release" \
        --notes "First release of Advanced Shape Detection and Coloring System

## Features
- Multiple shape detection algorithms
- Computer vision with OpenCV
- Machine learning integration
- Comprehensive documentation
- CI/CD pipeline with GitHub Actions
- Docker support
- Complete testing suite"'''
    
    result = run_command(cmd)
    
    if result and result.returncode == 0:
        print("✅ Initial release berhasil dibuat")
        return True
    else:
        print("⚠️ Gagal membuat release (bisa dibuat manual)")
        return False


def main():
    """Main function."""
    print("🚀 GitHub Setup Script untuk Advanced Shape Detection")
    print("=" * 60)
    
    # Cek apakah di dalam Git repository
    if not Path(".git").exists():
        print("❌ Bukan dalam Git repository")
        print("Jalankan: git init")
        return
    
    # Setup Git config
    if not check_git_config():
        if not setup_git_config():
            return
    
    # Setup GitHub CLI
    if not check_github_cli():
        if not install_github_cli():
            print("⚠️ Lanjutkan dengan manual setup")
            return
    
    # Login GitHub
    if not login_github():
        return
    
    # Buat repository
    if not create_repository():
        if not setup_remote():
            return
    
    # Push ke GitHub
    if not push_to_github():
        return
    
    # Setup tambahan
    setup_branch_protection()
    create_initial_release()
    
    print("\n🎉 Setup GitHub selesai!")
    print("\n📋 Langkah selanjutnya:")
    print("1. Buka repository di GitHub")
    print("2. Cek tab Actions untuk CI/CD pipeline")
    print("3. Cek tab Issues untuk template")
    print("4. Cek tab Pull requests untuk workflow")
    print("5. Update README.md dengan username Anda")
    
    print(f"\n🔗 Repository URL: https://github.com/YOUR_USERNAME/advanced-shape-detection")


if __name__ == "__main__":
    main()
