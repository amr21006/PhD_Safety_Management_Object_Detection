import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from textblob import TextBlob

# Sample responses for each category for sentiment analysis (based on previous thematic analysis)
responses = {
    "model_accuracy": [
        "Honestly, the model is very accurate in detecting workers who aren’t wearing helmets or reflective vests. It was able to identify many cases easily.",
        "The model is good at detecting helmets, but sometimes it makes mistakes when the vest is not clearly visible or the worker is turned around.",
        "I find the model’s accuracy excellent. I haven’t noticed any mistakes in identifying workers who aren’t adhering to PPE rules.",
        “The model is good, but it sometimes gets confused when there are multiple workers close to each other or when the lighting is poor”,
        "There were a few cases where the reflective vests weren’t very visible, but the model was able to correctly identify most workers not wearing helmets",
        "I didn’t notice any problems with the model’s accuracy. It can identify workers not wearing PPE clearly.",
        "The model is accurate in general, but it needs some improvement in cases of non-ideal lighting conditions",
        "I’m very impressed with the model’s accuracy, especially in crowded situations or when there is a lot of movement on the site",
        "The model is excellent in detecting helmets and vests. It was able to identify them even when the workers were far from the camera",
        "I think the model’s accuracy is very good and provides an effective means of monitoring worker compliance with PPE",
        "The model is excellent at identifying individuals entering hazardous zones. It alerts us quickly when someone crosses the safety line",
        "Sometimes it gets confused when the worker is near the boundary of the hazardous zone, but in general, it can identify them",
        "I think the model needs to be improved in defining hazardous zones more accurately. There were a few times when it didn’t send an alert.",
        "The model can identify individuals entering hazardous zones well, but we could add a feature to specify the type of hazard",
        "I am comfortable with the model’s performance in identifying hazardous zones. It provides an extra layer of protection for workers",
        "The model is accurate most of the time, but we need to make sure it can handle all types of hazardous zones",
        "I think the model can be very effective in preventing workers from entering prohibited areas.",
        "The model provides an excellent means of monitoring hazardous areas, but we need to connect it to the alarm system more effectively",
        "The model can handle different sizes of hazardous zones, which is an important advantage",
        "I’m optimistic that the model will help us improve worker safety by preventing them from entering dangerous areas."
    ],
    "alert_system_efficiency": [
        "The model is good at identifying workers who work at heights without harnesses. It was able to identify many cases.",
        "It sometimes makes mistakes in identifying wrong lifting techniques, but in general it can detect unsafe practices.",
        "I think the alert system needs some improvement. There were a few times when it didn’t send alerts on time",
        "The alerts were very helpful. I was able to easily identify the type of violation and its location, and intervene immediately",
        "I am comfortable with the speed and effectiveness of the alert system. It provides an important tool for immediate intervention in case of hazards.",
        "The alerts were clear and easy to understand. Anyone can understand them even if they are not tech-savvy",
        "We could add a feature to send alerts to mobile phones to ensure they reach the responsible people faster",
        "I think the alert system provides an effective way to alert us to violations, but we need to make sure it covers all the risks",
        "The alerts were accurate in most cases. I didn’t notice any errors in the type of violation or its location",
        "I’m optimistic that the alert system will help us reduce workplace accidents by intervening immediately in case of violations.",
        "Yes, the alerts provided us with enough information about the type of violation and its exact location. We were able to intervene quickly and appropriately",
        "Sometimes the information was not enough. For example, when someone wasn’t wearing a helmet, it didn’t specify the type of helmet required",
        "The alerts were clear and detailed. We were able to act correctly based on the information they contained",
        "We could add more information to the alerts, such as the name of the worker who violated the rule and the time of the incident",
        "I think the information in the alerts is enough for us to intervene appropriately. I didn’t notice any lack of information",
        "The alerts gave us information about the type of violation, but we could add pictures or videos to illustrate the violation",
        "I am comfortable with the information in the alerts. They provide everything we need to intervene quickly and effectively",
        "We could develop the alert system to be able to identify the severity of the violation and prioritize them accordingly",
        "The alerts were accurate and helpful. We were able to act correctly based on the information they contained",
        "I’m optimistic that the alert system will help us improve worker safety by providing enough information for appropriate intervention",
        "Yes, we had a few false alerts. This might make workers lose trust in the system and stop taking it seriously",
        "The false alerts were very rare, and they didn’t affect our trust in the system. We understand that any system can make mistakes sometimes",
        "I think false alerts are a big problem. We need to work on reducing them for the system to be effective and reliable.",
        "False alerts can reduce the system’s efficiency and waste the time of those responsible for verifying them",
        "I am comfortable with the accuracy of the alert system. I haven’t noticed any false alerts, which makes us confident in the system’s performance",
        "False alerts can create a sense of indifference among workers. They might start ignoring real alerts as well",
        "It’s very important to ensure the accuracy of the alert system and minimize errors so that the system is effective and accepted by workers",
        "False alerts can affect work productivity. The more they increase, the less focused the workers become.",
        "I think the alert system provides an effective way to alert us to violations, but we need to ensure its accuracy more.",
        "I’m optimistic that the alert system will be a powerful tool for enhancing safety on-site, but we need to address the issue of false alerts",
    ],
    "usability_and_interface": [
        "The interface is easy and simple. Anyone can use it without needing complex training.",
        "I feel the interface needs some improvement. There are a few things that are not clear or difficult to access."
        "The overall design of the interface is good, but we could add some more explanation for people who are not tech-savvy",
        "The interface is practical and gives us all the information we need. It is easy to use and very helpful",
        "I am comfortable with the ease of use of the interface. I was able to learn it quickly without needing help",
        "We could use different symbols or colors to make it easier to understand the information displayed in the interface",
        "I think the interface is suitable for users from different backgrounds. There is nothing complicated about it.",
        "We could add some interactive features to the interface, like the ability to zoom in on images or search for specific information",
        "The interface gives us a clear view of the site and the workers. I was able to monitor the situation easily through it",
        "I’m optimistic that the user-friendly interface will help us implement the system effectively on-site",
        "The data presentation is excellent. The colors are clear and help understand the information quickly",
        "I feel we could use different colors to better distinguish between types of hazards",
        "The interface is well-organized and provides clear information. It’s easy to access the information we need",
        "We could add some charts or graphs to better illustrate the data, especially if there is a lot of data",
        "I am comfortable with the data presentation. The colors are clear and the information is easy to understand",
        "We could use larger or clearer fonts to make it easier to read the information from a distance",
        "I think the data presentation is suitable for users from different backgrounds. There is nothing complicated about it",
        "We could add some interactive features, like the ability to filter the data or analyze it based on different criteria.",
        "The interface gives us a clear view of the site and the workers. I was able to monitor the situation easily through it",
        "I’m optimistic that the data presentation will help us analyze the information better and make sound decisions.",
        "Yes, the interface is simple and easy. Anyone can use it even if they are not familiar with technology",
        "We might need some basic training for workers who are not used to using software, but generally the interface is easy",
        "I think the interface is easy to understand, but we could add some instructions or explanatory videos",
        "The interface is easy to use and does not require much technical expertise. Anyone can learn it quickly",
        "I am comfortable with the ease of use of the interface. I was able to learn it quickly without needing help",
        "We could use symbols or pictures instead of words to make the interface easier to understand for workers who don’t read well",
        "I think the interface is suitable for all levels of technical expertise. There is nothing complicated about it",
        "We could create a simplified version of the interface for workers who don’t use technology much",
        "The interface gives us all the information we need in a clear and simple way. It doesn’t need any explanation",
        "I’m optimistic that the user-friendly interface will help us implement the system effectively across the entire site",
    ],
    "impact_on_safety_practices": [
        "We can use this model as part of our current safety monitoring system. It will help us identify hazards faster",
        "This model can be a powerful tool for safety training. We can show the videos to workers and explain to them how to avoid hazards",
        "I believe this model can replace manual inspection in certain situations. It will save a lot of time and effort",
        "We could link this model to the site’s alarm system. When a violation occurs, the alarm will trigger automatically",
        "I am comfortable with the idea of incorporating this model into our current safety management system. It will significantly help us improve our safety standards",
        "This model can be an important tool for analyzing site safety data. We can use this data to identify hazardous areas and create preventive plans",
        "It’s very important that we make sure all workers understand how to use this system correctly. We need comprehensive training",
        "This model can help improve communication between workers and those responsible for safety. It will help them exchange information faster and more accurately",
        "I am optimistic that this model will help us create a safer work environment for workers on construction sites.",
        "We can use this model as part of the regular safety risk assessment process at the site. It will help us make more informed decisions",
        "I’m sure this model will make workers more careful about their safety. When they know there is a system monitoring them, they will adhere to safety rules more",
        "This model can be an effective tool for educating workers about the importance of safety. It can show them videos that illustrate the dangers of not adhering to safety rules",
        "I think this model can create a sense of self-awareness among workers. They will know that their actions are being recorded and monitored, and this will encourage them to comply",
        "We can use this model to assess worker performance in terms of adherence to safety rules. We can reward workers who maintain their safety.",
        "I am comfortable with using this model to promote a safety culture at the site. It will make workers more careful about their safety and the safety of their colleagues",
        "It’s important that we clarify to workers that the purpose of this model is not to spy on them, but to help them stay safe",
        "I think this model will help us reduce workplace accidents by increasing worker awareness of the importance of safety",
        "We can integrate this model with safety training programs. It will help workers apply what they have learned in real-world situations",
        "I am optimistic that this model will help us build better relationships between workers and those responsible for safety. It will help them cooperate better",
        "We can use this model to evaluate the effectiveness of safety training programs. We can measure their impact on worker behavior",
        "Absolutely, this model will help us prevent accidents before they happen. It can identify risks early on, giving us a chance to take action before any harm occurs",
        "This model can be a powerful tool for predicting risks. It can analyze data and identify areas or activities where the likelihood of accidents is higher",
        "I think this model will help us make more proactive decisions regarding safety. It will give us accurate information so we can plan better",
        "We can use this model to simulate different scenarios and see how we can prevent accidents in each scenario",
        "I am comfortable with the idea of using this model to",
        "It is important that we integrate this model with the current risk management system. It will help us identify and assess risks more accurately",
        "I believe this model will change our way of thinking about safety on construction sites. It will make us think more proactively.",
        "This model can be a valuable tool for raising risk awareness among workers. When they see the system identifying risks in real time, they will be more careful",
        "I am optimistic that this model will help us reduce the cost of accidents by preventing them from happening in the first place",
        "We can use this model to monitor the contractors’ compliance with safety standards. It can be linked to a contractor performance evaluation system",
    ],
    "challenges_and_suggestions": [
        "The biggest challenge will be the cost of the system. It might be a little expensive for some contracting companies",
        "We need to make sure the system will work efficiently in all weather conditions. For example, if it’s raining or dusty, will it affect the system’s performance",
        "Some workers might reject the idea of constant surveillance. We need to explain the importance of the system to them and ensure their privacy",
        "We need comprehensive training for workers and supervisors on how to use the system. Everyone must understand how to use it correctly",
        "We need regular maintenance of the system to ensure it is working efficiently. There must be a dedicated team for system maintenance",
        "There might be a challenge in integrating this system with existing systems at the site. We need a thorough study of system compatibility",
        "There might be a challenge in continuously updating the system to keep pace with technological advancements. We need a clear plan for system updates",
        "It is important that we regularly measure the effectiveness of the system. There must be clear indicators for measuring system performance",
        "There might be a challenge in providing the necessary technical support for the system. We need a dedicated technical support team",
        "What suggestions do you have for improving the model’s performance, functionality, or user interface? Are there any specific features or capabilities that you think would be beneficial",
        "We could add the feature of sending alerts to mobile phones to ensure they reach the responsible people faster",
        "We can improve the accuracy of the system by training it on images and videos from different construction sites",
        "We could add data analysis features to identify high-risk areas and create preventive plans",
        "We could link this system to the human resource management system so we can monitor the workers’ performance in terms of compliance with safety rules",
        "We could add a feature that automatically records videos when a violation occurs. This will help us analyze errors better",
        "We could improve the interface to make it more user-friendly. We could use symbols or pictures instead of words",
        "We could add a feature to generate periodic reports on system performance. This will help us evaluate the effectiveness of the system",
        "We could add remote control functionality to the system. This will help us monitor the site from anywhere",
        "We could use artificial intelligence techniques to enhance the system’s ability to predict risks",
        "It is important that we ensure that the system is easy to update. There must be an easy way to install new updates",
    ]
}

# --- Sentiment Analysis ---
sentiments = {}
for category, texts in responses.items():
    category_sentiments = []
    for text in texts:
        blob = TextBlob(text)
        category_sentiments.append(blob.sentiment.polarity)
    sentiments[category] = category_sentiments

def categorize_sentiments(sentiment_scores):
    summary = {“positive”: 0, “neutral”: 0, “negative”: 0}
    for score in sentiment_scores:
        if score > 0.1:
            summary[“positive”] += 1
        elif score < -0.1:
            summary[“negative”] += 1
        else:
            summary[“neutral”] += 1
    return summary

summary_results = {domain: categorize_sentiments(scores) for domain, scores in sentiments.items()}

# --- Visualization Functions ---
def create_sentiment_pie_chart(sentiments, category, title):
    positive, neutral, negative = calculate_sentiment_distribution(sentiments, category)
    labels = [“Positive”, “Neutral”, “Negative”]
    sizes = [positive, neutral, negative]
    colors = [“lightgreen”, “lightblue”, “lightcoral”]

    # Filter out zero values to avoid errors
    filtered_labels = [label for label, size in zip(labels, sizes) if size > 0]
    filtered_sizes = [size for size in sizes if size > 0]
    filtered_colors = [color for color, size in zip(colors, sizes) if size > 0]

    plt.figure(figsize=(6, 6))  # Adjust Figure size as needed
    plt.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, autopct=’%1.1f%%’, startangle=90,  wedgeprops={“linewidth”: 1, “edgecolor”: “white”})
    plt.axis(“equal”)  
    plt.title(title, pad=20) # Add padding to the title
    plt.show()

def calculate_sentiment_distribution(sentiments, category):
    positive = sum(1 for s in sentiments[category] if s > 0.1)
    negative = sum(1 for s in sentiments[category] if s < -0.1)
    neutral = len(sentiments[category]) - positive - negative
    return positive, neutral, negative

def create_wordcloud(texts, title):
    text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5)) # Wider Figure for better display
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, pad=20) # Larger title and padding
    plt.tight_layout(pad=0) 
    plt.show()

# --- Generate Visualizations ---

for category, category_sentiments in sentiments.items():
    title = f"Sentiment Analysis: {category.replace('_', ' ').title()}"
    create_sentiment_pie_chart(sentiments, category, title)

for category, texts in responses.items():
    title = f"Word Cloud: {category.replace('_', ' ').title()}"
    create_wordcloud(texts, title)

# Overall Word Cloud
all_texts = [text for texts in responses.values() for text in texts]
create_wordcloud(all_texts, "Overall Word Cloud of Expert Feedback")
 
