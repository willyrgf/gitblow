{
  description = "GitBlow distributed by Nix.";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          matplotlib
          gitpython
        ]);
      in
      {
        packages.gitblow = pkgs.stdenv.mkDerivation rec {
          pname = "gitblow";
          version = "0.0.2";
          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];
          buildInputs = [ pythonEnv pkgs.git ];

          dontBuild = true;

          installPhase = ''
            echo "Contents of source directory:"
            ls -la
            mkdir -p $out/bin $out/lib
            if [ -f "gitblow.py" ]; then
              cp gitblow.py $out/lib/
              makeWrapper ${pythonEnv}/bin/python $out/bin/gitblow \
                --add-flags "$out/lib/gitblow.py" \
                --prefix PATH : ${pkgs.git}/bin
            else
              echo "ERROR: gitblow.py not found, adding it..."
              exit 1
            fi
          '';

          shellHook = ''
            export PATH=${pythonEnv}/bin:${pkgs.git}/bin:$PATH
          '';
        };

        defaultPackage = self.packages.${system}.gitblow;

        defaultApp = {
          type = "app";
          program = "${self.packages.${system}.gitblow}/bin/gitblow";
        };
      });
}
