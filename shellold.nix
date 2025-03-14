# shell.nix

let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.numpy
      python-pkgs.matplotlib
      python-pkgs.pandas
      python-pkgs.ipykernel
      python-pkgs.jupyterlab
    ]))
  ];
}