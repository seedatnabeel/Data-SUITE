papermill notebooks/electric_pipeline.ipynb results/electric_pipeline.ipynb --log-output --log-level INFO
papermill notebooks/electric_artefacts_analysis.ipynb results/electric_artefacts_analysis.ipynb --log-output --log-level INFO

rm -rf results/electric_pipeline.ipynb
rm -rf results/electric_artefacts_analysis.ipynb
