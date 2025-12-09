# Generated Problem Directory Overview

> **Warning**
> During the development of this project, CVRPLIB updated some of their databases and links. Consequently, some scripts relying on specific URL structures or file formats might not function as expected without modification.

This document explains the purpose of each file and folder in the `generated_problem` directory.

## Python Files

### scripts_gen/generate_sh.py
This script generate `.sh` containing argument for uchoa instance generation.

### generate_pb.py
This script automates the generation of problem instances and organizes them into folders.  
- **Process**:  
  - Searches for `.sh` scripts in the `scripts_gen` folder.  
  - Creates a corresponding folder in `bdd_generated` for each script.  
  - Executes the script to generate problem instances.  
  - Moves `.vrp` files (problem instances) into the appropriate folder.  

### generator.py
This script contains the logic for generating problem instances based on specific parameters.  
- **Purpose**: Used by `generate_pb.py` to create `.vrp` files.  

### create_torch_bdd.py
This scripts transform `.vrp` into torch tensor and save them


This file was developed by:

- **Uchoa et al. (2017)**: *New benchmark instances for the Capacitated Vehicle Routing Problem*. European Journal of Operational Research.  
- **Queiroga, Eduardo, et al. (2022)**: *10,000 optimal CVRP solutions for testing machine learning-based heuristics*.  

For more details about the generation process, refer to these publications.

## Folders

### bdd_generated
This folder contains subdirectories named `gen<number>` (e.g., `gen1`, `gen2`), `<number>` being the number of clients.  
- **Contents**:  
  - Each subdirectory corresponds to a generated problem set.  
  - Includes `.vrp` files (problem instances) and other related files.

### scripts_gen
This folder contains `.sh` scripts used for generating problem instances.  
- **Purpose**: Each script defines specific parameters for generating a set of problems.
