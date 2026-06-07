## A. General Project Requirements
- The project must combine at least two of the following blocks:
    - ML Numeric Data
    - NLP
    - Computer Vision

- The selected blocks must be meaningfully integrated (conceptually and technically)
- The project must use multiple and different data sources.
- The project must use data sources which were not used during the semester (apartment from canton of zurich and dogs breed for image). 
- Well-motivated and realistic use case
- The project must be completed independently and properly documented.

**Note:** Detailed block-specific requirements can be found below in Section E.

## B. Documentation Requirements (Mandatory)
You must fully complete the provided project documentation template, including:

1. Project Idea & Methodology
- Clear problem definition and objectives
- Well-motivated and realistic use case
- Explanation of how the selected blocks are combined
- Clear scope and assumptions

2. Data & Preprocessing
- Description of all data sources (type, origin, size)
- Data cleaning and preparation steps
- Block-specific preprocessing (e.g. tokenization, normalization, image preprocessing)
- Feature engineering, annotation, or augmentation (if applicable)
- Exploratory data analysis (EDA) with key findings

3. Modeling & Implementation
- Justification of model or prompt selection
- Training, fine-tuning, or development strategy
- Comparison of models, prompts, or approaches
- Iterations and improvements
- Technical implementation details (libraries, structure)

4. Evaluation & Analysis
- Clear evaluation strategy (data splits, metrics, qualitative analysis)
- Model performance analysis
- Error analysis
- Interpretation of results
- Block-specific evaluation (e.g. qualitative NLP outputs, visual CV results)

5. Deployment
- A working deployment (URL required)
- Separation of training and inference
- Screenshots demonstrating key functionality

6. Execution Instructions
- Clear instructions on how to run the project locally or reproduce results

## C. Assessment Criteria
- All projects are graded using a uniform rubric, independent of the chosen block combination.
- Evaluation focuses on:
    - Clarity
    - Technical correctness
    - Depth of analysis
    - Quality of integration
    - Clean and reproducible implementation
- **Bonus points** may be awarded for optional extensions (e.g. all three blocks, extended evaluation, ethical considerations).

## D. Submission
The project and its documentation must be submitted as github link by **07 June 2026, 18:00.**

**Note:** You must add the following github to your project

- Jasmin Heierli -jasminh
- Benjamin Kühnis - bkuehnis

## E. Specific Block Requirements
This section describes the **minimum requirements for each AI block when used as part of a combined project.**

Each project must integrate **at least two of the three blocks** listed below.
The requirements define the **expected contribution of each selected block** within a **single, coherent AI application.**

***Important:***
- The requirements below do **not** describe standalone projects.
- Each block must be implemented in relation to the other selected block(s) and contribute meaningfully to the overall system.

### General Rules for Combined Blocks
- Each selected block must be:
    - clearly identifiable in the project
    - technically and conceptually integrated into the overall application
    - clearly identifiable in the project and evaluated appropriately within the single, unified project documentation
- Blocks must interact through:
    - shared data
    - derived features
    - model outputs
    - decision logic
    - or user interaction
- Simply executing multiple independent models side-by-side is **not sufficient.**
- The selected blocks do not need to contribute equally in size or complexity, but each must provide a **clear and justified contribution.**

### E.1 ML Numeric Data (Machine Learning on Structured Data)
If the project includes the **ML Numeric Data** block, it must demonstrate the following:

- Use of at least one **structured or numeric dataset**
- Exploratory Data Analysis (EDA) to understand data distributions, relationships, or anomalies
- Feature engineering, feature selection, or transformation
- Training and comparison of **at least two different models**
- Quantitative evaluation using appropriate metrics
- Interpretation of results and error analysis
- Clear explanation of how the numeric model:
    - uses outputs from another block **or**
    - provides inputs or decisions to another block

### E.2 NLP (Natural Language Processing / LLMs / RAG)
If the project includes the **NLP** block, it must demonstrate the following:

- Clear definition of the text-based data (e.g. documents, conversations, prompts, user input)
- NLP-specific preprocessing and/or prompt design
- Use of at least one NLP approach, such as:
    - classical NLP models
    - transformer-based models
    - retrieval-augmented generation
    - prompt engineering
- Comparison of models, prompts, or retrieval strategies (at least one comparison)
- Appropriate qualitative and/or quantitative evaluation
- Explanation of how the NLP component:
    - enhances interpretation, interaction, or decision-making
    - integrates with another block (e.g. explaining numeric predictions, guiding user interaction)

### E.3 Computer Vision
If the project includes the **Computer Vision** block, it must demonstrate the following:

- Use of image data (collected or existing datasets)
- Image preprocessing and/or data augmentation
- Training, fine-tuning, or application of at least one vision model
- Evaluation using suitable metrics and/or visual inspection
- Interpretation of model behavior and limitations
- Explanation of how visual information:
    - generates features for another block **or**
    - is interpreted, explained, or acted upon by another block

### Examples of Valid Block Combinations
The following examples illustrate possible (non-exhaustive) combinations:

- **ML Numeric Data + NLP**
    - Numeric prediction combined with natural language explanation or interaction.
- **ML Numeric Data + Computer Vision**
    - Image-derived features used as inputs for a numeric prediction model.
- **NLP + Computer Vision**
    - Multimodal system combining text interaction with image understanding.
- **ML Numeric Data + NLP + Computer Vision**
    - Fully integrated multimodal application (eligible for bonus points).

### Documenation Template
Under following link you find the documentation template you must use https://github.com/bkuehnis/ai-applications/blob/main/project/documentation_template.md.

***Don't change the structure of the template.***

**Check the "Documentation Hint" on how to refer to your code when documenting your project.**