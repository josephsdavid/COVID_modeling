{lib, buildPythonPackage, fetchPypi}:

buildPythonPackage rec {
    pname = "cord-19-tools";
    version = "0.1.0";

    src = fetchPypi {
      inherit pname version;
      sha256 = "8b47959363b65ac4d9cde04d7a4c20a9c0a48de1cdf51df2985932bb8dfe3036";
    };
    doCheck = false;
}
