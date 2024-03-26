{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.nltk
    pkgs.python3Packages.transformers 
    pkgs.python3Packages.sentencepiece
    
    pkgs.python3Packages.numpy
    pkgs.python3Packages.torch
    pkgs.python3Packages.requests
  ];
}