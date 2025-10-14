
# 🛍️ E-Commerce Product Recommender  

An intelligent, **behavior-driven product recommendation system** that combines **content-based filtering**, **user behavior analytics**, and **LLM-powered explanations** to generate personalized product suggestions for e-commerce users.  

Built with **Streamlit**, **Scikit-learn**, and **Google Gemini API**, this project demonstrates how AI can enhance online shopping experiences with both **smart recommendations** and **human-like explanations**.

---

## 🚀 Features  

✅ **Behavior-driven personalization**  
- Learns from user actions such as product views and purchases.  
- Gives higher priority to categories and products the user interacts with most.  

✅ **Content-based similarity**  
- Uses TF-IDF (Term Frequency–Inverse Document Frequency) on product descriptions to compute product similarity.  

✅ **AI-powered explanations**  
- Integrates Google Gemini (Generative AI) to explain *why* a product is recommended in natural language.  

✅ **Auto image filling**  
- Automatically assigns category-specific product images for a polished, complete recommendation display.  

✅ **Interactive Streamlit dashboard**  
- Upload CSVs, select users, and visualize recommendations dynamically.  

---

## 🧠 Tech Stack  

| Component | Technology Used |
|------------|-----------------|
| **Frontend** | Streamlit |
| **Backend / Logic** | Python, Pandas, Scikit-learn |
| **AI/LLM Integration** | Google Gemini API |
| **Vectorization** | TF-IDF (Content-based filtering) |
| **Data Input** | Product & User Behavior CSVs |

---

## 📂 Project Structure  

```

E-Commerce-Product-Recommender/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Dependencies
├── sample_data/
│   ├── product.csv            # Example product dataset
│   ├── user_behavior.csv      # Example behavior dataset
│
├── README.md                  # Project documentation
└── .env (optional)            # GEMINI_API_KEY for AI explanations

````

---

## 🧾 Input CSV Format  

### **1. product.csv**
| id | name | category | description | price | image_url (optional) |
|----|------|-----------|--------------|--------|-----------------------|
| p1 | Comfort Running Shoes | Footwear | Lightweight shoes for daily runs. | 59.99 | *(optional)* |

### **2. user_behavior.csv**
| user_id | product_id | event |
|----------|-------------|--------|
| u1 | p1 | view |
| u1 | p3 | purchase |

---

## ⚙️ Installation & Setup  

1. **Clone the repository**

   git clone [GitHub Repository](https://github.com/Shi1709/E-Commerce-Product-Recommender/tree/main)
   cd E-Commerce-Product-Recommender


2. **Install dependencies**

   pip install -r requirements.txt


3. **(Optional) Set your Gemini API key**
 
   export GEMINI_API_KEY="your_api_key_here"
   

4. **Run the app**

   streamlit run app.py



## 🧩 How It Works

1. Upload the **product catalog** and **user behavior** CSV files.
2. The system:

   * Builds a **TF-IDF similarity matrix** for product descriptions.
   * Analyzes the user’s viewing and purchasing behavior.
   * Combines both to create a personalized product ranking.
3. Optionally, Gemini generates natural-language “Why we recommend this” explanations.
4. View and download the recommendations with product images.
