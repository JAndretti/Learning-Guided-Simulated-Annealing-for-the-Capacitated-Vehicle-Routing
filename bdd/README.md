# CVRP Databases

> **Warning**
> During the development of this project, CVRPLIB updated some of their databases and links. Consequently, some scripts relying on specific URL structures or file formats might not function as expected without modification.

This folder contains various databases related to the Capacitated Vehicle Routing Problem (CVRP). These databases include problem instances and their corresponding solutions.

The databases are sourced from [CVRPLIB](http://vrp.galgos.inf.puc-rio.br/index.php/en/), a well-known repository for CVRP instances and solutions.

## Downloading the Databases

To download the databases, run the `settings.py` file located at the root of the project. This script will handle the setup and download of the required files.

## Explanation of `generate_csv_data.py`

The `generate_csv_data.py` script processes the set **Vrp-Set-XML100**, to generate a CSV file containing the following information:
- **Instance Name**: The name of the CVRP instance.
- **Optimal Solution Cost**: The cost of the optimal solution for the instance.
- **OR-Tools Solution Cost**: The cost computed using the OR-Tools solver.

The script reads CVRP instance files, extracts relevant data (e.g., node coordinates, demands, capacities), and uses the OR-Tools solver to compute solution costs. It then compares these costs with the optimal solution costs provided in the database. The results are saved in a CSV file for further analysis.
