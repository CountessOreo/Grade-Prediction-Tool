# MLG382-Guided-Project

**BrightPath Academy Student Success Project**

 **About BrightPath Academy**
    
    Located in a bustling urban area, BrightPath Academy is a progressive secondary school. 
    The school encourages not only academic achievement but also extracurricular activity and personal development because it is dedicated to both academic excellence and holistic development.

  **Mission**
    
    To ensure that every learner achieves their full potential by empowering them via early academic interventions and individualized instruction.

  **BrightPath's Problems**
  
   The academy faces several significant obstacles in spite of their progressive stance:

    1.  Delayed Identification of At-Risk Students: Academically disadvantaged students are not promptly identified.
    2. Absence of Targeted Support: Tutoring and mentorship are examples of interventions that aren't customized to meet the requirements of specific students.
    3. Limited Extracurricular Activity Insights: Not enough information is available to assess the impact of extracurricular activity on academic performance.
    4. Data Overload: Teachers collect a lot of data, but they don't have a consolidated, actionable method to use when making judgments.

  **Problem Statement**
    
    BrightPath Academy is commited to academic and personal excellence, it however is struggling to proactively support students who may fall behind. 
    The system that is currently in use lacks a timely mechanism that identifies learners who are at risk, which leads to the failure of being able to identify oppurtunites for intervention.

    The interventions are also not unique and are more generic, which serves as a problem as different students will require different needs. The educators also do not have access to a tool that will allow them to process the large volumes of data. 

    This project aims to leverage machine learning to not only be able to predict students who are at risk, but to also uncover patterns that will provide personalized and effective support strategies for the students. 
    This will assist the academy to make data-driven decisions that will enhance the academic outcomes, optimize resource allocation and to improve the well-beng of the students.

 ## **Hypothesis**
  
**First Analysis:**

From the univariate and bivariate analysis graphs, we observed that absenteeism has a clear negative correlation with academic performance. Students with higher absenteeism tend to be in lower grade classes, as seen in the scatter plots and correlation coefficients. On the other hand, study time shows a positive relationship with grade classification, where students who study more weekly tend to be in higher grade classes. The relationship between parental support and tutoring and grade classification also appears to be positive, with higher levels of support correlating with better academic outcomes.

Hypothesis:
Based on the analysis, we predict that as absenteeism increases, the likelihood of being placed in a lower grade class also increases. This aligns with the trend observed in the data, where students with higher absenteeism generally have lower grades. However, we expect that students who dedicate more time to studying each week may be classified into higher grade classes despite having a higher absenteeism rate, indicating that study habits can counterbalance absenteeism.


**Second Analysis:**

The graphs also show that students receiving parental support and tutoring tend to be in higher grade classes, suggesting that academic support can mitigate the effects of absenteeism. Parental education levels appear to have a slight positive correlation with grade classification, though the effect is not as pronounced. Extracurricular activities, like sports, music, and volunteering, show a weak to moderate positive relationship with grade classification.

Hypothesis:
We predict that students who receive parental support and tutoring will have a higher likelihood of being placed in higher grade classes, regardless of their absenteeism rate. These support mechanisms are expected to buffer against the negative impact of absenteeism on academic performance. Additionally, extracurricular activities may improve academic outcomes, though their effect is expected to be secondary to academic support systems like tutoring and parental involvement.

**Third Analysis:**

In the analysis, parental education level seems to have a moderate impact on academic outcomes, with students from families with higher parental education levels being more likely to be classified into higher grade classes. Gender, age, and ethnicity show minimal correlation with grade classification, suggesting that these factors are less significant in predicting academic performance when compared to study time, absenteeism, and parental support.

Hypothesis:
We hypothesize that parental education level will positively correlate with student performance, as more educated parents may offer better academic guidance. However, we do not expect gender, age, or ethnicity to have a strong influence on the model’s predictions, as they are less tied to academic behaviors and support structures.

 **Final Prediction:**
 
By evaluating the feature importance scores across the models, we expect absenteeism to have the greatest negative influence on grade classification, while tutoring will have the greatest positive influence. Study time and parental support will also be important factors, with tutoring and parental support buffering against the negative effects of absenteeism. Features like age, ethnicity, and gender are likely to have the least impact on the model's predictions.

  **Key Outcomes**
    
    1. Early detection of at-risk students
    2. Actionable insights for educators
    3. Understanding of how extracurriculars impact academic performance
    4. Centralized, intuitive dashboard for student success strategies

