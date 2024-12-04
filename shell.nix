let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
    packages = [
    (pkgs.python3.withPackages (python-pkgs: [
        python-pkgs.pip
        python-pkgs.ipython
        python-pkgs.jupyter
        python-pkgs.notebook
        python-pkgs.ipykernel
        python-pkgs.pandas
        python-pkgs.numpy
        python-pkgs.seaborn
        python-pkgs.matplotlib
        python-pkgs.scipy
    ]))
  ];
}