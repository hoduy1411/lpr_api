# compile code
nuitka \
--follow-import-to=app.src \
--include-package=app.src \
--follow-import-to=app.api \
--include-package=app.api \
--follow-import-to=app.core \
--include-package=app.core \
--follow-import-to=app.main \
--include-package=app.main \
--follow-import-to=models \
--include-package=models \
--enable-plugins=upx \
--static-libpython=no \
--remove-output \
run.py

# clean code
rm -rf /code/models && \
rm -rf /code/run.py && \
rm -rf /code/build.sh && \
rm -rf /code/app/api && \
rm -rf /code/app/core && \
rm -rf /code/app/main && \
rm -rf /code/app/src && \
rm -rf /code/app/model/*/*.pt