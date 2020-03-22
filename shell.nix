let
  pkgs = import <stable> {};
  #stable = import <stable>{};

  cord19 =  pkgs.callPackage ./cord.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    fetchPypi = pkgs.python37.pkgs.fetchPypi;
  };


in
  pkgs.mkShell {
    name = "rona";
    buildInputs = [
      pkgs.python37
      cord19
      pkgs.python37Packages.pymc3
      pkgs.python37Packages.plotly
      pkgs.python37Packages.numpy
      pkgs.python37Packages.dask
      pkgs.python37Packages.scikitlearn
      pkgs.python37Packages.beautifulsoup4
      pkgs.python37Packages.xgboost
      pkgs.zip
      pkgs.python37Packages.scipy
      pkgs.python37Packages.matplotlib
      pkgs.python37Packages.seaborn
      pkgs.python37Packages.jupyter
      pkgs.python37Packages.pandas
      pkgs.python37Packages.imbalanced-learn
      pkgs.qt5.qtbase
      pkgs.python37Packages.wheel
      pkgs.python37Packages.twine
    ];
    shellHook = ''
      export SOURCE_DATE_EPOCH=$(date +%s) 
    '';

  }
