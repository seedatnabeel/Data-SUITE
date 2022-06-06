papermill notebooks/adult_pipeline.ipynb results/adult_pipeline.ipynb --log-output --log-level INFO
papermill notebooks/adult_artefacts_analysis.ipynb results/adult_artefacts_analysis.ipynb --log-output --log-level INFO

rm -rf results/adult_artefacts_analysis.ipynb
rm -rf results/adult_pipeline.ipynb
