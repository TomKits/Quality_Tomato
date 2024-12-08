import tensorflow as tf  
import numpy as np  
import cv2  
import matplotlib.pyplot as plt  


model = tf.keras.models.load_model('E:/KULIAHAHAHAH/semester 5 bismillah/Studi Independet Bersertifikat (Bangkit)/project/code/New folder/efficientnetb3-QualityTomatoes-99.13.h5')  

def preprocess_image(img_path):  
    img_size = (224, 224)  
    img = cv2.imread(img_path)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, img_size)  
    img = np.expand_dims(img, axis=0)  # Add batch dimension  
    return img  

def get_class_names_above_threshold(predictions, threshold=0.7):  
    class_names = [  
        'Bad',  
        'Ripe',  
        'Unripe',  
    ]  
    selected_classes = []  
    for idx, score in enumerate(predictions):  
        if score >= threshold:  
            selected_classes.append((class_names[idx], score * 100))  # Append class name and confidence score  
    return selected_classes if selected_classes else [("No class above threshold", 0)]  

def predict_image(img_path, threshold=0.7):  
    image_preproses = preprocess_image(img_path)  
    prediction = model.predict(image_preproses)[0]  # Get prediction for the image (single batch)  
    
    # Get class names above the threshold  
    classes_above_threshold = get_class_names_above_threshold(prediction, threshold=threshold)  
    
    # Load and display the image  
    img = cv2.imread(img_path)  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    plt.imshow(img_rgb)  
    
    if classes_above_threshold[0][0] != "No class above threshold":  
        display_text = ", ".join([f"{class_name} ({confidence_score:.2f}%)"  
                                  for class_name, confidence_score in classes_above_threshold])  
    else:  
        display_text = "No class above threshold"  
    
    plt.title(display_text)  
    plt.axis('off')  
    plt.show()  

    print(f"Predictions: {prediction}")  
    for class_name, confidence_score in classes_above_threshold:  
        print(f"Hasil Prediksi: {class_name}, Confidence: {confidence_score:.2f}%")  
    
    return classes_above_threshold  

# List of image paths  
img_paths = [  
    # 'E:/KULIAHAHAHAH/semester 5 bismillah/Studi Independet Bersertifikat (Bangkit)/project/code/New folder/busuk_buah-300x227.jpg',
    'E:/KULIAHAHAHAH/semester 5 bismillah/Studi Independet Bersertifikat (Bangkit)/project/code/New folder/download.jpeg' 
    # Add other image paths here  
]  

# Predict for each image  
for img_path in img_paths:  
    predict_image(img_path, threshold=0.7)