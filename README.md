# CS-552 - Milestone 2

Welcome to the next step of your MNLP project! For M2, as you can read in the [project description](https://docs.google.com/document/d/1SP8SCHPOZZGEhs2ay-38FjedRE1bS9Q99VJb28eHoYk/edit?usp=sharing), you have 3 main deliverables: 
1. Your full dataset
2. Your model
3. The progress report


## Repo Structure

The repo has 5 folders, 3 of which serve for you to submit the deliverables:
1. `_templates` contains the latex template for both your progress report. You MUST use these templates.
2. `_tests` contains some scripts which run automated tests so you can be sure your submission is correctly formatted (e.g., that the right files exist in the right place). **Importantly, if your team is NOT implementing RAG, you should change the first line of `_tests/model_rag_valitor.py` into `IMPLEMENTING_RAG = False`.**
	- Note: we test the documents directory exists for the groups who will be implementing RAG, but **you don't need to implement RAG for this milestone!** Since you should submit your full data for M2, however, we expect that you will already have populated your documents folder, if you are implementing RAG.
3. `data` should be populated with your full dataset, following the format of [the examples in the datasets folder of the project code template folder](https://github.com/CS-552/project-code-2024/tree/main/datasets).
4. `model` should contain your model, using the [provided code template](https://github.com/CS-552/project-code-2024). You MUST use this template.
5. `pdfs` should contain a single pdf file, your progress report (named `<YOUR-GROUP-NAME>.pdf`).

## Running the tests manually
The autograding tests run automatically with every commit to the repo. Please check M1's repo for instructions on running the tests manually if you wish to do so.