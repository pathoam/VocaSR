#!/usr/bin/env python3
"""
Comprehensive packaging script for Voca
Creates various distribution formats for easy installation
"""

import os
import sys
import json
import shutil
import tarfile
import zipfile
import platform
import subprocess
from pathlib import Path
from datetime import datetime


class VocaPackager:
    """Main packaging coordinator for Voca distributions."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.absolute()
        self.dist_dir = self.project_root / "dist"
        self.version = self._get_version()
        self.platform_info = self._get_platform_info()
        
    def _get_version(self) -> str:
        """Extract version from pyproject.toml or default."""
        try:
            pyproject = self.project_root / "pyproject.toml"
            if pyproject.exists():
                with open(pyproject, 'r') as f:
                    for line in f:
                        if line.startswith("version"):
                            return line.split('"')[1]
        except:
            pass
        return "1.0.0"
    
    def _get_platform_info(self) -> dict:
        """Get current platform information."""
        system = platform.system().lower()
        arch = platform.machine().lower()
        
        return {
            "system": system,
            "arch": arch,
            "exe_name": "voca.exe" if system == "windows" else "voca",
            "dist_name": f"voca-{system}-{arch}",
            "archive_ext": ".zip" if system == "windows" else ".tar.gz"
        }
    
    def create_portable_package(self):
        """Create a portable package with executable and all necessary files."""
        print("\n📦 Creating Portable Package...")
        
        dist_name = f"{self.platform_info['dist_name']}-v{self.version}"
        package_dir = self.dist_dir / dist_name
        
        # Clean and create package directory
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True)
        
        # Copy executable
        exe_path = self.dist_dir / self.platform_info['exe_name']
        if not exe_path.exists():
            print("❌ Executable not found. Run 'make build' first.")
            return False
        
        shutil.copy2(exe_path, package_dir / self.platform_info['exe_name'])
        
        # Create portable launcher script
        if self.platform_info['system'] == 'windows':
            launcher = package_dir / "run_voca.bat"
            launcher.write_text("""@echo off
echo Starting Voca...
"%~dp0voca.exe" %*
""")
        else:
            launcher = package_dir / "run_voca.sh"
            launcher.write_text("""#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/voca" "$@"
""")
            launcher.chmod(0o755)
        
        # Add README for portable version
        readme = package_dir / "README_PORTABLE.txt"
        readme.write_text(f"""Voca Portable v{self.version}
=====================================

This is a portable version of Voca that can be run from any location.

QUICK START:
{'- Double-click run_voca.bat' if self.platform_info['system'] == 'windows' else '- Run: ./run_voca.sh'}
- Or run the voca executable directly

NO INSTALLATION REQUIRED!
Just extract this folder anywhere and run.

FIRST RUN:
- Press {self._get_default_hotkey()} to start/stop recording
- Configuration will be created in your user directory
- Change settings by editing the config file

For more information, visit the project repository.
""")
        
        # Create archive
        archive_name = f"{dist_name}-portable{self.platform_info['archive_ext']}"
        archive_path = self.dist_dir / archive_name
        
        if self.platform_info['system'] == 'windows':
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file in package_dir.rglob('*'):
                    zf.write(file, file.relative_to(package_dir.parent))
        else:
            with tarfile.open(archive_path, 'w:gz') as tf:
                tf.add(package_dir, arcname=dist_name)
        
        print(f"✅ Portable package: {archive_path}")
        return True
    
    def create_installer_package(self):
        """Create an installer package with setup scripts."""
        print("\n🔧 Creating Installer Package...")
        
        dist_name = f"{self.platform_info['dist_name']}-v{self.version}-installer"
        package_dir = self.dist_dir / dist_name
        
        # Clean and create package directory
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True)
        
        # Copy executable
        exe_path = self.dist_dir / self.platform_info['exe_name']
        if not exe_path.exists():
            print("❌ Executable not found. Run 'make build' first.")
            return False
        
        shutil.copy2(exe_path, package_dir / self.platform_info['exe_name'])
        
        # Create platform-specific installer
        if self.platform_info['system'] == 'windows':
            self._create_windows_installer(package_dir)
        elif self.platform_info['system'] == 'darwin':
            self._create_macos_installer(package_dir)
        else:
            self._create_linux_installer(package_dir)
        
        # Create archive
        archive_name = f"{dist_name}{self.platform_info['archive_ext']}"
        archive_path = self.dist_dir / archive_name
        
        if self.platform_info['system'] == 'windows':
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file in package_dir.rglob('*'):
                    zf.write(file, file.relative_to(package_dir.parent))
        else:
            with tarfile.open(archive_path, 'w:gz') as tf:
                tf.add(package_dir, arcname=dist_name)
        
        print(f"✅ Installer package: {archive_path}")
        return True
    
    def _create_windows_installer(self, package_dir: Path):
        """Create Windows installer scripts."""
        # Install script
        install_script = package_dir / "install.bat"
        install_script.write_text(f"""@echo off
echo ====================================
echo Voca Installer v{self.version}
echo ====================================
echo.

set INSTALL_DIR=%LOCALAPPDATA%\\Voca
set START_MENU=%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs

echo Installing to: %INSTALL_DIR%
echo.

:: Create installation directory
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

:: Copy executable
echo Copying files...
copy /Y voca.exe "%INSTALL_DIR%\\" >nul

:: Create Start Menu shortcut (using PowerShell)
echo Creating Start Menu shortcut...
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%START_MENU%\\Voca.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\\voca.exe'; $Shortcut.IconLocation = '%INSTALL_DIR%\\voca.exe'; $Shortcut.Save()"

:: Add to PATH (optional, requires admin)
echo.
echo Installation complete!
echo.
echo Voca has been installed to: %INSTALL_DIR%
echo You can now:
echo   - Run Voca from the Start Menu
echo   - Or run directly: %INSTALL_DIR%\\voca.exe
echo.
echo Press any key to exit...
pause >nul
""")
        
        # Uninstall script
        uninstall_script = package_dir / "uninstall.bat"
        uninstall_script.write_text("""@echo off
echo ====================================
echo Voca Uninstaller
echo ====================================
echo.

set INSTALL_DIR=%LOCALAPPDATA%\\Voca
set START_MENU=%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs
set CONFIG_DIR=%APPDATA%\\Voca

echo This will uninstall Voca from your system.
echo.
set /p CONFIRM=Are you sure? (y/N): 
if /i not "%CONFIRM%"=="y" exit

:: Remove executable and directory
if exist "%INSTALL_DIR%" (
    echo Removing installation files...
    rmdir /S /Q "%INSTALL_DIR%"
)

:: Remove Start Menu shortcut
if exist "%START_MENU%\\Voca.lnk" (
    echo Removing Start Menu shortcut...
    del "%START_MENU%\\Voca.lnk"
)

:: Ask about config files
if exist "%CONFIG_DIR%" (
    echo.
    set /p REMOVE_CONFIG=Remove configuration files? (y/N): 
    if /i "%REMOVE_CONFIG%"=="y" (
        echo Removing configuration...
        rmdir /S /Q "%CONFIG_DIR%"
    )
)

echo.
echo Voca has been uninstalled.
echo Press any key to exit...
pause >nul
""")
    
    def _create_macos_installer(self, package_dir: Path):
        """Create macOS installer scripts."""
        install_script = package_dir / "install.sh"
        install_script.write_text(f"""#!/bin/bash

echo "===================================="
echo "Voca Installer v{self.version}"
echo "===================================="
echo

INSTALL_DIR="/Applications/Voca"
BIN_LINK="/usr/local/bin/voca"

echo "Installing to: $INSTALL_DIR"
echo

# Create installation directory
sudo mkdir -p "$INSTALL_DIR"

# Copy executable
echo "Copying files..."
sudo cp voca "$INSTALL_DIR/"
sudo chmod +x "$INSTALL_DIR/voca"

# Create symlink for command line access
echo "Creating command line link..."
sudo ln -sf "$INSTALL_DIR/voca" "$BIN_LINK"

# Create .app bundle (basic)
echo "Creating application bundle..."
APP_DIR="$INSTALL_DIR/Voca.app/Contents/MacOS"
sudo mkdir -p "$APP_DIR"
sudo cp voca "$APP_DIR/"
sudo chmod +x "$APP_DIR/voca"

# Create Info.plist
sudo tee "$INSTALL_DIR/Voca.app/Contents/Info.plist" > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Voca</string>
    <key>CFBundleExecutable</key>
    <string>voca</string>
    <key>CFBundleIdentifier</key>
    <string>com.voca.app</string>
    <key>CFBundleVersion</key>
    <string>{self.version}</string>
</dict>
</plist>
EOF

echo
echo "Installation complete!"
echo
echo "You can now:"
echo "  - Run Voca from Applications folder"
echo "  - Or use 'voca' command in Terminal"
echo
echo "Note: You may need to grant accessibility permissions"
echo "in System Preferences > Security & Privacy > Privacy"
""")
        install_script.chmod(0o755)
        
        # Uninstall script
        uninstall_script = package_dir / "uninstall.sh"
        uninstall_script.write_text("""#!/bin/bash

echo "===================================="
echo "Voca Uninstaller"
echo "===================================="
echo

INSTALL_DIR="/Applications/Voca"
BIN_LINK="/usr/local/bin/voca"
CONFIG_DIR="$HOME/Library/Application Support/Voca"

echo "This will uninstall Voca from your system."
read -p "Are you sure? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    exit 0
fi

# Remove installation
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing installation files..."
    sudo rm -rf "$INSTALL_DIR"
fi

# Remove symlink
if [ -L "$BIN_LINK" ]; then
    echo "Removing command line link..."
    sudo rm "$BIN_LINK"
fi

# Ask about config files
if [ -d "$CONFIG_DIR" ]; then
    echo
    read -p "Remove configuration files? (y/N): " REMOVE_CONFIG
    if [[ "$REMOVE_CONFIG" =~ ^[Yy]$ ]]; then
        echo "Removing configuration..."
        rm -rf "$CONFIG_DIR"
    fi
fi

echo
echo "Voca has been uninstalled."
""")
        uninstall_script.chmod(0o755)
    
    def _create_linux_installer(self, package_dir: Path):
        """Create Linux installer scripts."""
        install_script = package_dir / "install.sh"
        install_script.write_text(f"""#!/bin/bash

echo "===================================="
echo "Voca Installer v{self.version}"
echo "===================================="
echo

# Detect if running as root
if [ "$EUID" -eq 0 ]; then
    INSTALL_DIR="/opt/voca"
    BIN_DIR="/usr/local/bin"
    DESKTOP_DIR="/usr/share/applications"
    SYSTEM_INSTALL=true
else
    INSTALL_DIR="$HOME/.local/opt/voca"
    BIN_DIR="$HOME/.local/bin"
    DESKTOP_DIR="$HOME/.local/share/applications"
    SYSTEM_INSTALL=false
fi

echo "Installation type: $([ "$SYSTEM_INSTALL" = true ] && echo "System-wide" || echo "User")"
echo "Installing to: $INSTALL_DIR"
echo

# Create directories
mkdir -p "$INSTALL_DIR" "$BIN_DIR" "$DESKTOP_DIR"

# Copy executable
echo "Copying files..."
cp voca "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/voca"

# Create symlink
echo "Creating command line link..."
ln -sf "$INSTALL_DIR/voca" "$BIN_DIR/voca"

# Create desktop entry
echo "Creating desktop entry..."
cat > "$DESKTOP_DIR/voca.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Voca
Comment=Voice-to-Text Transcription Tool
Exec=$INSTALL_DIR/voca
Icon=audio-input-microphone
Terminal=true
Categories=Utility;AudioVideo;
EOF

# Update desktop database if available
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null
fi

# Add to PATH if needed
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo
    echo "Note: Add $BIN_DIR to your PATH to run 'voca' from anywhere:"
    echo "  echo 'export PATH=\\$PATH:$BIN_DIR' >> ~/.bashrc"
fi

echo
echo "Installation complete!"
echo
echo "You can now:"
echo "  - Run 'voca' from terminal"
echo "  - Launch from your application menu"
echo "  - Or run directly: $INSTALL_DIR/voca"
""")
        install_script.chmod(0o755)
        
        # Uninstall script
        uninstall_script = package_dir / "uninstall.sh"
        uninstall_script.write_text("""#!/bin/bash

echo "===================================="
echo "Voca Uninstaller"
echo "===================================="
echo

# Detect installation type
if [ -d "/opt/voca" ] && [ "$EUID" -eq 0 ]; then
    INSTALL_DIR="/opt/voca"
    BIN_DIR="/usr/local/bin"
    DESKTOP_DIR="/usr/share/applications"
elif [ -d "$HOME/.local/opt/voca" ]; then
    INSTALL_DIR="$HOME/.local/opt/voca"
    BIN_DIR="$HOME/.local/bin"
    DESKTOP_DIR="$HOME/.local/share/applications"
else
    echo "Voca installation not found."
    exit 1
fi

CONFIG_DIR="$HOME/.config/Voca"

echo "This will uninstall Voca from your system."
read -p "Are you sure? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    exit 0
fi

# Remove installation
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing installation files..."
    rm -rf "$INSTALL_DIR"
fi

# Remove symlink
if [ -L "$BIN_DIR/voca" ]; then
    echo "Removing command line link..."
    rm "$BIN_DIR/voca"
fi

# Remove desktop entry
if [ -f "$DESKTOP_DIR/voca.desktop" ]; then
    echo "Removing desktop entry..."
    rm "$DESKTOP_DIR/voca.desktop"
    
    # Update desktop database if available
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$DESKTOP_DIR" 2>/dev/null
    fi
fi

# Ask about config files
if [ -d "$CONFIG_DIR" ]; then
    echo
    read -p "Remove configuration files? (y/N): " REMOVE_CONFIG
    if [[ "$REMOVE_CONFIG" =~ ^[Yy]$ ]]; then
        echo "Removing configuration..."
        rm -rf "$CONFIG_DIR"
    fi
fi

echo
echo "Voca has been uninstalled."
""")
        uninstall_script.chmod(0o755)
    
    def create_debian_package(self):
        """Create a .deb package for Debian/Ubuntu systems."""
        if self.platform_info['system'] != 'linux':
            print("⚠️  Debian package can only be created on Linux")
            return False
        
        print("\n📦 Creating Debian Package...")
        
        deb_name = f"voca_{self.version}_{self.platform_info['arch']}"
        deb_dir = self.dist_dir / deb_name
        
        # Clean and create package structure
        if deb_dir.exists():
            shutil.rmtree(deb_dir)
        
        # Create Debian package structure
        (deb_dir / "DEBIAN").mkdir(parents=True)
        (deb_dir / "usr" / "local" / "bin").mkdir(parents=True)
        (deb_dir / "usr" / "share" / "applications").mkdir(parents=True)
        
        # Copy executable
        exe_path = self.dist_dir / "voca"
        if not exe_path.exists():
            print("❌ Executable not found. Run 'make build' first.")
            return False
        
        shutil.copy2(exe_path, deb_dir / "usr" / "local" / "bin" / "voca")
        
        # Create control file
        control = deb_dir / "DEBIAN" / "control"
        control.write_text(f"""Package: voca
Version: {self.version}
Section: utils
Priority: optional
Architecture: {self.platform_info['arch']}
Maintainer: Voca Team
Description: Voice-to-Text Transcription Tool
 A cross-platform voice-to-text transcription application
 that allows users to quickly transcribe speech using
 hotkey activation.
""")
        
        # Create desktop entry
        desktop = deb_dir / "usr" / "share" / "applications" / "voca.desktop"
        desktop.write_text("""[Desktop Entry]
Version=1.0
Type=Application
Name=Voca
Comment=Voice-to-Text Transcription Tool
Exec=/usr/local/bin/voca
Icon=audio-input-microphone
Terminal=true
Categories=Utility;AudioVideo;
""")
        
        # Build .deb package
        try:
            subprocess.run(
                ["dpkg-deb", "--build", str(deb_dir)],
                check=True,
                capture_output=True
            )
            print(f"✅ Debian package: {deb_dir}.deb")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create .deb package: {e}")
            return False
        except FileNotFoundError:
            print("⚠️  dpkg-deb not found. Install with: sudo apt install dpkg")
            return False
    
    def create_appimage(self):
        """Create an AppImage for universal Linux distribution."""
        if self.platform_info['system'] != 'linux':
            print("⚠️  AppImage can only be created on Linux")
            return False
        
        print("\n📦 Creating AppImage...")
        print("⚠️  AppImage creation requires additional tools.")
        print("   Consider using portable package instead.")
        return False
    
    def _get_default_hotkey(self) -> str:
        """Get the default hotkey for the platform."""
        if self.platform_info['system'] == 'darwin':
            return "Cmd+Shift+V"
        else:
            return "Ctrl+Shift+V"
    
    def create_all_packages(self):
        """Create all applicable packages for the current platform."""
        print(f"\n🚀 Creating all packages for {self.platform_info['system']} ({self.platform_info['arch']})")
        print(f"   Version: {self.version}")
        
        results = []
        
        # Always create portable and installer packages
        results.append(("Portable", self.create_portable_package()))
        results.append(("Installer", self.create_installer_package()))
        
        # Platform-specific packages
        if self.platform_info['system'] == 'linux':
            results.append(("Debian", self.create_debian_package()))
        
        # Summary
        print("\n" + "="*50)
        print("📊 Packaging Summary:")
        for name, success in results:
            status = "✅" if success else "❌"
            print(f"  {status} {name} Package")
        
        print(f"\nAll packages saved to: {self.dist_dir}/")
        print("\nRecommended distribution:")
        print("  - Portable: For users who want no installation")
        print("  - Installer: For users who want proper system integration")
        if self.platform_info['system'] == 'linux':
            print("  - Debian: For Ubuntu/Debian users (apt install)")


def main():
    """Main entry point for packaging script."""
    packager = VocaPackager()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "portable":
            packager.create_portable_package()
        elif sys.argv[1] == "installer":
            packager.create_installer_package()
        elif sys.argv[1] == "deb" and packager.platform_info['system'] == 'linux':
            packager.create_debian_package()
        elif sys.argv[1] == "--help":
            print("Usage: python package.py [portable|installer|deb|all]")
            print("  portable  - Create portable package (no installation)")
            print("  installer - Create installer package")
            print("  deb       - Create Debian package (Linux only)")
            print("  all       - Create all applicable packages (default)")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Run with --help for usage")
    else:
        packager.create_all_packages()


if __name__ == "__main__":
    main()