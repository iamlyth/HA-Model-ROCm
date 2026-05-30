{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.gcc
    pkgs.python312
  ];
  buildInputs = [
    pkgs.python312
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];
  shellHook = ''
    source .venv/bin/activate

    # Dynamically find Python.h in nix store
    PY_INCLUDE=$(find /nix/store -name "Python.h" -path "*3.12*" 2>/dev/null \
      | grep -v source | head -1 | xargs dirname 2>/dev/null)
    if [ -n "$PY_INCLUDE" ]; then
      export C_INCLUDE_PATH="$PY_INCLUDE:$C_INCLUDE_PATH"
      export CPATH="$PY_INCLUDE:$CPATH"
    fi

    # Add ROCm SDK libs from venv (gfx1151 nightly)
    VENV_SITE="$PWD/.venv/lib/python3.12/site-packages"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VENV_SITE/_rocm_sdk_core/lib:$VENV_SITE/_rocm_sdk_libraries_gfx1151/lib"
    export NIX_LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH:$VENV_SITE/_rocm_sdk_core/lib:$VENV_SITE/_rocm_sdk_libraries_gfx1151/lib"

    export PYTORCH_ALLOC_CONF=expandable_segments:True
  '';
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    pkgs.zlib
    pkgs.python312
  ];
  NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    pkgs.zlib
    pkgs.python312
  ];
  NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
}
