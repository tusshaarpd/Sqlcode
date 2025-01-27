# Sqlcode

Input:

User provides a plain English description of the SQL query they need.
Example: "Get the average salary of employees grouped by department for the last 5 years."
Processing:

The app uses the mrm8488/t5-small-finetuned-wikiSQL model, an open-source model fine-tuned for generating SQL queries.
The input is tokenized and passed to the model, which generates an SQL query.
Output:

The generated SQL query is displayed in a formatted code block.
Example Output:
sql
Copy
Edit
SELECT department, AVG(salary)
FROM employees
WHERE date >= '2018-01-01'
GROUP BY department;
