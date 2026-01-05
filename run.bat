@echo off

pushd "%~dp0"
git pull --quiet
"%~dp0venv\Scripts\python.exe" -m manga_translator %*
popd
