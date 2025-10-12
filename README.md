# Pepper_Prices_Analysis 



👩‍💻 I'm currently working on...

---

## *Author  [Omar Soub](https://github.com/omars1234)*

## *Overview*

*On this Project ,we will intensive data analysis techneques for Bell Pepper data set Using Python Programming*

*This Project provides the industry with a well reliable analysis that drive the decision making in the industry of Bell Pepper*


* *Bell Pepper -->*<img src="Capture.PNG" alt="Logo" width="200" style="vertical-align:middle; margin-right:10px;">


---
---

## *Project :*


```bash
git clone https://github.com/omars1234/Pepper_Prices_Analysis.git
```

```bash
conda create --name EnvPepperPricesAnalysis python=3.10 -y
```

```bash
conda activate EnvPepperPricesAnalysis
```

```bash
pip install -r requirements.txt
```


## *Project Structure :*


*1. Data Exploration*

   * *A. Data Description :*

     * *The historical dataset (actual.csv) ,has the following Features:*
     
       * *categorical_cols :['p_color']*
     
       * *numerical_cols : ['vietnam_season', 'price', 'total_volume', 'brazil', 'india', 'vietnam', 'indonesia', 'china', 'jordan_max_price', 'jordan_min_price', 'demand', 'supply']*

       * *boolean_cols : ['brazil_season', 'indonesia_season', 'india_season', 'china_season']*

       * *date_cols : ['week_start_dt', 'week_end_dt']*


*2. Data Cleaning*

* *1. Remove the rows with value -1*  
*2. Na Values check*  
*3. Fiiling Na vlaues using interplolate with linear moethod*  
*4. Convert week_start_dt & week_end_dt to datetime datatype*  
*5. Detect numerical_cols,categorical_cols,boolean_cols, and date_cols*  

*3. Feature Engeneering*

* *Create new features from existing ones :*  
   * year  
   * month  
   * year
   * and DayOfMonth

*4. Exploratory Data Analysis-EDA*   

* *1. have a quick look at the Number of unique values in each feature*  
* *2. numerical_cols EDA : distribution ,basic statistic summary,correlation and visualization*  
* *3. categorical_cols EDA : distribution ,basic statistic summary and visualization*  
* *4. boolean_cols EDA : distribution ,basic statistic summary and visualization*  

*5. Probabilities*  

* *1. Empirical probability*  
* *2. Distribution probability*

*6. Inferential statistics*  
* *1. Mann-Whitney U test --> non-parametric statistical test used when we have non-normaly distributed target feature and Binary inpute Feature*

* *2. Spearman's rank correlation test --> non-parametric statistical test used when we have non-normaly distributed target feature and Ordinal inpute Feature to test the monotonic relationship*

* *3. Kruskal-Wallis test--> non-parametric statistical test used when we have non-normaly distributed target feature and Categorical inpute Feature then have more than 2 categories to*










 ----------------------------------------

## *Feedback*

*If you have any feedback, please reach out to us at omars.soub@gmail.com*

## 🔗 Links

[*my github page-https://github.com/omars1234*](https://github.com/omars1234)

## *🛠 Skills*
*python, R, SQL ,PowerBi ,Tableaue*