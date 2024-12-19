# AcademiQ: A Computer Science Research Search Engine

AcademiQ is a specialized search engine that helps researchers, students, and institutions find computer science-related papers, authors, and research institutions using **Learning to Rank** techniques.

---

## 🚀 **Project Overview**
The goal is to build a search engine that:
- Identifies **scholars** and **research institutions** for computer science topics.
- Optimizes search relevancy using **LambdaMART Learning to Rank (L2R)** models.

---

## 🛠️ **Core Features**
1. **Data Integration**:
   - Collects data from **Google Scholar API**, **Scopus API**, **arXiv**, and **DBLP**.
2. **Machine Learning**:
   - Implements **LambdaMART (L2R)** for search ranking.
   - Supports **LightGBM** for fast and efficient training.
3. **User Interface**:
   - Built with **Flask** (backend) and **Bootstrap** (frontend) for user-friendly interaction.

---

## 🧩 **Project Structure**

```plaintext
AcademiQ/
├── dataset/            # Data ingestion and preprocessing
├── files/              # LambdaMART and LightGBM code
├── static/             # Project logo and CSS/Bootstrap of template
├── templates/          # HTML template of user interface
├── main.py             # Backend server using Flask
└── README.md           # Project documentation
```

## ⚙️ **Technical Details**

### Data Collection
- **APIs**: Google Scholar, Scopus, arXiv, DBLP.
- **Data Includes**:
   - **Papers** (title, abstract, citations).
   - **Authors** (name, affiliation, citation count).
   - **Institutions** (organization and rankings).

### Learning to Rank (L2R)
- **Algorithm**: LambdaMART (RankNet + NDCG).
- **Framework**: LightGBM.
- **Training**:
   - Features extracted include:
     - TF-IDF, BM25, PageRank, HITS (Hub & Authority scores), Citation count, etc.

### Evaluation
- **Metric**: NDCG@K (Normalized Discounted Cumulative Gain).

---

## 🖥️ **Deployment**
- **Backend**: Deployed on **AWS EC2**.
- **Frontend**: Flask handles UI rendering.
