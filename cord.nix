{lib, buildPythonPackage, fetchPypi}:

buildPythonPackage rec {
    pname = "cord-19-tools";
    version = "0.0.8";

    src = fetchPypi {
      inherit pname version;
      sha256 = "38b313dabd09ee42a16afcb763029b383bf7d2cabcdb82f820f5bf3eaa7762ee";
    };
    doCheck = false;
}
