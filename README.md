Research Trend Visualization Framework
# Overview
This project presents a visualization framework for identifying research trends within university departments, focusing on Computer Science (CS) and Computational Science and Engineering (CSE) faculty at Georgia Tech. By using clustering algorithms and network graphs, the framework uncovers patterns, overlaps, and collaborations among faculty research areas.

# Key Features

- **Research Clustering:** Groups faculty based on research interests using clustering techniques.
- **Network Graphs:** Visualizes relationships and interdisciplinary connections.
- **Pattern Analysis:** Identifies key thematic areas and emerging trends.
- **Reusable Framework:** Can be adapted for other academic departments.

# Methodology
1. **Data Collection**
  In this project, we used Beautiful Soup, a Python library, to scrape faculty profiles from the CS department website. Beautiful Soup allowed us to parse HTML pages, identify specific elements such as research interests and affiliations, and systematically extract this information. This was essential for compiling a comprehensive dataset of faculty research areas, which served as the foundation for subsequent data cleaning, clustering, and visualization.
2. **Text Analysis with TF-IDF**
  To extract meaningful features from the research interests, we used TF-IDF, a popular technique in text analysis.
3. **K-Means Clustering**
  We applied K-means clustering to the feature set derived from the TF-IDF method. We initially experimented with 10 clusters to capture a broader range of research topics. However, after evaluating the clustering results, we found that reducing the number of clusters to 5 yielded more meaningful and cohesive groupings of faculty with similar research interests.
4. **Network Graph**
   For the network graph analysis, we treated faculty members as nodes and their research interests as edges connecting those nodes. Before establishing these connections, we first identified unique research labels from the data to ensure that each faculty member was connected to the appropriate research areas. Once the unique labels were identified, we linked the faculty members to their corresponding research topics.

# Tech Stack
- Programming: Python
- Data Processing: Pandas, Scikit-learn, Pytorch
- Visualization: NetworkX, Matplotlib, Seaborn, Plotly
   
  
  



