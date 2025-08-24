
# PRUEBA TÉCNICA
Para la realización de la prueba, se utilizó Jupiter Notebook, siéntase libre de ejecutar los notebooks en local, sin embargo se recomienda usar un entorno en la nube como Google Colab, ya que ofrece procesamiento gráfico gratuito.

La solución está estructurada en las siguientes secciones o archivos.

- Clasificador *(./classifier)*
  - Generador de datos sintéticos *(classifier_sintetic_data_creation.ipynb)*
  - Entrenamiento del modelo *(trainModel_roberta-large-bne-finetuned-mldoc.ipynb)*
  - Inferencia *(useModel_roberta-large-bne-finetuned-mldoc.ipynb)*
- Reconocimiento de entidades nombradas *(./ner)*
  - Generador de datos sintéticos *(ner_sintetic_data_creation.ipynb)*
  - Entrenamiento del modelo *(trainModel_bert-spanish-cased-finetuned-ner.ipynb)*
  - Inferencia *(useModel_bert-spanish-cased-finetuned-ner.ipynb)*

En las respectivas carpetas también encontrará los archivos con los datos en crudo utilizados para entrenar a los modelos.

**NOTAS IMPORTANTES:** 
- No es necesario entrenar los modelos para realizar predicciones, siéntase libre de ejecutar los archivos de inferencias, ya que los modelos entrenados se encuentran alojados en Hugging Face listos para ser utilizados.
- Para la generación de datos sintéticos, es necesario la utilización de API KEYS de DeepSeek, las cuales las puede obtener en el siguiente enlace  
    [Gestionar API Keys de DeepSeek](https://platform.deepseek.com/api_keys)
- Los datos sintéticos fueron revisados manualmente para evitar ruido en el entrenamiento, 
  - Para el modelo clasificador se utilizó excel.
  - Para el modelo NER, se utilizó label-studio: Para ello primero se generaron los datos sintéticos en formato JSON, posteriormente se utilizó label-studio para revisar las etiquetas, y con esta herramienta se exportó a formato conll.
    - A pesar de que no se va a tratar el uso de label-studio, se dejará el archivo de configuracion de etiquetas, por si se llega a utilizar de alguna manera.
    ```html
        <View>
            <Labels name="label" toName="text">
                <Label value="PERSONA" background="#a600ff"/>
                <Label value="CARGO" background="#0011ff"/>
                <Label value="DINERO" background="#ff9500"/>
                <Label value="FECHA" background="#000000"/>
                <Label value="PROYECTO" background="#ff0000"/>
                <Label value="REGLAMENTO" background="#4dff00"/>
            </Labels>

            <Text name="text" value="$text"/>
        </View>
    ```
- Puede encontrar más información sobre Google Colab aquí: [Uso de Colab](https://colab.research.google.com)

## CLASIFICADOR

En la siguiente sección se explica el entrenamiento y utilización de un modelo de clasificacion basado en BERT.

### ***Cómo ejecutar***
Instalar las bibliotecas necesarias y reiniciar el entorno una vez instaladas.
```bash
!pip install transformers datasets tensorflow huggingface_hub -q > /dev/null 2>&1
!pip install "numpy<2.0"
```


En los notebooks, el algoritmo para estructurar los datos necesarios para el modelo se encuentra comentado, sin embargo se puede acceder al dataset ya configurado mediante Hugging Face ejecutando la celda. Siéntase libre de descomentar y jugar con el código, solo necesita cargar el archivo ***dataTraining.csv*** en el entorno o en google drive en caso de utilizar Google Colab.

```python
#Cargar los datos ya seteados desde el repo (En caso de no poseer el .csv)
from datasets import load_dataset
dataset = load_dataset("aatituanav/roberta-base-bne-mldoc-4cat")
dataset
```

La estructura del dataset ya divide los datos en train, val, test

```
DatasetDict({
    train: Dataset({
        features: ['body', 'label', 'idx', '__index_level_0__'],
        num_rows: 320
    })
    validation: Dataset({
        features: ['body', 'label', 'idx', '__index_level_0__'],
        num_rows: 40
    })
    test: Dataset({
        features: ['body', 'label', 'idx', '__index_level_0__'],
        num_rows: 40
    })
})
```
Para el entrenamiento, se carga el tokenizador, se crea la función tokenizadora y se utiliza para tokenizar los datos 
```python
#Se carga el tokenizador
model_name = "dccuchile/roberta-base-bne-finetuned-mldoc"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["body"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

# Aplicar tokenización
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Elimina las columnas de texto originales
tokenized_datasets = tokenized_datasets.remove_columns(["body"])
print("Columnas después de tokenizar:", tokenized_datasets["train"].column_names)


# ========== [Preparación de Datasets] ==========
# Convertir a formato TensorFlow
train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=8
)

val_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=8
)

test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=8
)
```

La configuración del modelo clasificador se la realiza utilizando el cuerpo pre-entrenado de ***dccuchile/roberta-base-bne-finetuned-mldoc***, y se le contruye una cabeza personalizada de clasificación para 4 etiquetas, (Este enfoque permite escalar a más etiquetas sin esfuerzo)
```python
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

model = TFAutoModel.from_pretrained(model_name, from_pt=True, name="roberta_base")

#Construcción de la cabeza de clasificación
input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

# Salida del modelo base (pasando todas las entradas)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
pooled_output = outputs.last_hidden_state[:, 0, :]
dropout = tf.keras.layers.Dropout(0.7)(pooled_output)
classifier = tf.keras.layers.Dense(
    4,
    activation="softmax",
    kernel_regularizer=tf.keras.regularizers.l2(0.01)
)(dropout)
#Definicion del modelo completo
model = tf.keras.Model(
    inputs=[input_ids, attention_mask],
    outputs=classifier,
    name="roberta_base_bne_finetuned_mldoc"
)

#Compilación
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()
```
Se configura una detención anticipada, la cual monitorizará la pérdida en la validación y detendrá el entrenamiento automaticamente, esto permite obtener los pesos en la época en donde el modelo tenga mejor rendimiento. 

Posterioremente se ajusta el modelo
```python
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint  
#Parado temprano
early_stopping_acc = EarlyStopping(
    min_delta=0.0001,
    patience=3,
    restore_best_weights=True,
    verbose=1,
    monitor="val_loss",
    mode="min"
)

#Ajuste del modelo
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping_acc],
)
```
Las celdas posteriores en el notebook muestran distintas métricas del modelo. Sientase en la libertad de ejecutarlas consecutivamente.


### **Desiciones Técnicas**
- Utilización del modelo *dccuchile/roberta-base-bne-finetuned-mldoc* como modelo base ya que es una versión optimizada y eficiente del modelo ROBERTA, con un proceso de fine-tuning para la clasificación de documentos, con el dataset mldoc. Perfecto para este problema.

- **max_length=256 para tokenización:** A pesar de que el modelo admite 512, se reduce a la mitad para ahorro de memoria, ya que las resoluciones tienden a ser cortas.
- **Learning Rate: 1e-5:** Se utiliza un lr menor que el modelo NER, ya que con 2e-5 convergía demasiado rápido. (Aún así este valor permite que el modelo converja rápido).
- **Uso de EarlyStopping:** Permite la detención del modelo al analizar la pérdida en la validacion, se tiene una paciencia de 3, ya que al converger rápido, un valor menor aquí es la mejor opcion (también puede ser 2).
- **Epocas 10:** Al ser un modelo que converge sumamente rápido (También depende del set de datos que no tiene mucha variabilidad), se establece un número reducido de épocas. 
- **Métricas de rendimiento:** Para medir el rendimiento del modelo, se calculan métricas como exactitud en el conjunto de prueba, matriz de confusión y reporte de clasificación por clases, las cuales son un estándar para medir modelos de clasificación.


### **Realizar Inferencias**
Para ello cargue el archivo ***useModel_roberta-large-bne-finetuned-mldoc.ipynb*** en el entorno Jupyter, y ejecute la celda para instalar las bibliotecas necesarias.

La celda para cargar el modelo la tendrá mas abajo, hará todo el trabajo automáticamente.
```python
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from transformers import AutoTokenizer, TFRobertaModel
from huggingface_hub import snapshot_download

repo_id = "aatituanav/roberta-base-bne-mldoc-4cat"

# descarga de el repositorio completo
local_dir = snapshot_download(repo_id=repo_id)

# carga el modelo
model = tf.keras.models.load_model(
    local_dir,
    custom_objects={"TFRobertaModel": TFRobertaModel}
)

# carga el tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_dir)
```

Previamente, se define una función que permite hacer las predicciones 
```python
def predecir_texto(texto, tokenizer, model, max_length=256):
    # Tokenizar el texto de entrada
    inputs = tokenizer(
        texto,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="tf"
    )

    # Hacer la predicción
    logits = model.predict({
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })

    # Clase predicha
    clase_predicha = tf.argmax(logits, axis=1).numpy()[0]

    # Probabilidades de cada clase
    probabilidades = tf.nn.softmax(logits, axis=1).numpy()[0]

    return clase_predicha, probabilidades
```
Listo! Una vez cargado el pipeline, solo ejecútelo con el texto de su preferencia 

```python
texto_ejemplo = """Artículo 1.- Aprobación del Cronograma para Elecciones de Representantes
Estudiantiles del Instituto Superior Tecnológico Yaruquí.  """

# Predecir
clase, probs = predecir_texto(texto_ejemplo, tokenizer, model)

clases = {
    "0": "Infraestructura y Recursos",
    "1": "Nombramientos de Personal",
    "2": "Aprobaciones de Planes",
    "3": "Modificaciones Reglamentarias"
}

print("Resultado de la predicción:")
print(f"Texto: {texto_ejemplo}")
print(f"Clase predicha: {clase} - {clases[str(clase)]}")
print(f"Probabilidades:")
for i, prob in enumerate(probs):
    print(f"   Clase {i} ({clases[str(i)]}): {prob:.4f}")
```

SALIDA ESPERADA 
```
1/1 [==============================] - 0s 41ms/step
Resultado de la predicción:
Texto: Artículo 1.- Aprobación del Cronograma para Elecciones de Representantes
Estudiantiles del Instituto Superior Tecnológico Yaruquí.  
Clase predicha: 2 - Aprobaciones de Planes
Probabilidades:
   Clase 0 (Infraestructura y Recursos): 0.1822
   Clase 1 (Nombramientos de Personal): 0.1888
   Clase 2 (Aprobaciones de Planes): 0.4442
   Clase 3 (Modificaciones Reglamentarias): 0.1848
   
```
## RECONOCIMIENTO DE ENTIDADES NOMBRADAS
Este proyecto implementa un modelo de Reconocimiento de Entidades Nombradas (NER) utilizando una arquitectura basada en Transformers para el idioma español. El modelo es capaz de identificar y clasificar entidades como personas, cargos, cantidades de dinero, fechas, proyectos y referencias a reglamentos en textos en español.


### **Cómo Ejecutar**
Se ejecuta cada celda consecutivamente con las teclas Shift+ENTER

Instalación de dependencias
```bash
!pip install transformers tokenizers seqeval evaluate datasets==3.6.0 -q > /dev/null 2>&1
```
En los notebooks, el algoritmo para estructurar los datos necesarios para el modelo se encuentra comentado, sin embargo se puede acceder al dataset ya configurado mediante Hugging Face ejecutando la celda.
```python
from datasets import load_dataset
raw_datasets = load_dataset("aatituanav/bert-spanish-cased-finetuned-ner-6ent")
raw_datasets
```
Para evaluar las métricas del modelo NER se utilizó **wandb** al momento de ejecutar la siguiente celda 
```python
from transformers import Trainer

trainer = Trainer(
    model=model_for_training,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
trainer.train()
```
El cual va a pedir un código de autorización con el siguiente mensaje 

*"wandb: You can find your API key in your browser here: https://wandb.ai/authorize?ref=models"*

Se deberá abrir el link, previamente creada una cuenta en wanbd, aparecerá un token el cual deberá ser pegado en el cuadro de texto debajo del mensaje. Mismo que permitirá mostar métricas del entrenamiento del modelo en tiempo real. 

Las métricas para esta solucion las puede observar aquí

[![Vista previa](https://img.shields.io/badge/Ver_Sitio_Web-00a2ed?style=for-the-badge)](https://api.wandb.ai/links/carloswringo-universidad-central-del-ecuador/4yxxpkqs)

Estas son las consideraciones que hay que tomar al momento de ejecutar las celdas en orden, aquí unicamente se contempla el entrenamiento del modelo y la visualización de métricas de presición al final, por lo que no se guardará el modelo una vez ejecutado todo.

### **Desiciones Técnicas**
- Utilización del modelo *dccuchile/distilbert-base-spanish-uncased-finetuned-ner* como modelo base ya que es una versión optimizada y eficiente del modelo BERT y al ser pre-entrenado por la universidad de chile con un corpus en español, es perfecto para esta tarea.
- Se entrena al modelo con los siguentes labels solicitados en la prueba con sus respectivos (Begin-Inside-Outside)
```python
label_names = [
    'O',
    'B-PERSONA', 'I-PERSONA',
    'B-CARGO', 'I-CARGO',
    'B-DINERO', 'I-DINERO',
    'B-FECHA', 'I-FECHA',
    'B-PROYECTO', 'I-PROYECTO',
    'B-REGLAMENTO', 'I-REGLAMENTO'
]
```
- Para la evaluación se utiliza ***seqeval*** la cual permite analizar modelos entrenados para la clasificación de tokens, y al momento de calcular la presición en cada época se utiliza los siguientes métricas 
```python
return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
```
la cual mide presición de las entidades correctamente detectadas (La etiqueta de la entidad y su posición, si alguna de las dos falla, se da la entidad como mal detectada)
 - Se hace uso adicionalmente del cálculo de matriz de presición, la cual permite un análisis más fino por Entidad, permitiendo saber qué entidades le cuesta más al modelo detectar.
 - Hiperparámetros: 
    - **70% Entrenamiento - 15% Validación - 15% Prueba:** Se intentó con 80% para entrenamiento, sin embargo como el modelo tenía generalmente una buena presición, se optó por asignar más registros para medir mejor el desempeño.
    - **Learning Rate: 2e-5:** Estándar para fine-tuning de modelos Transformer, el cual permite ajustes precisos sin borrar el conocimiento pre-entrenado del modelo.
    - **Epocas 15:** A pesar de entrenar durante 15 epocas, el modelo mostró su mejor rendimiento en la época 8, por lo cual se tomó los pesos de esa época.

### **Realizar Inferencias**
Para ello cargue el archivo ***useModel_bert-spanish-cased-finetuned-ner.ipynb*** en el entorno Jupyter, y ejecute la celda para instalar las bibliotecas necesarias.

La celda para cargar el modelo la tendrá mas abajo, hará todo el trabajo automáticamente.
```python
from transformers import pipeline

#Se carga el modelo entrenado previamente, se puede colocar el modelo guardado
model_checkpoint = "aatituanav/bert-spanish-cased-finetuned-ner-6ent"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
```
Listo! Una vez cargado el pipeline, solo ejecútelo con el texto de su preferencia 
```
texto = """
Que, mediante resolución RCT-SE-04-Nro.58-2023 con fecha 16 de septiembre de 2023,
en sesión extraordinaria de Consejo Transitorio se posesionó a los representantes
docentes al Órgano Colegiado Superior del Instituto Superior Tecnológico Yaruqui.
Que, mediante resolución RCT-SE-04- Nro.60- 2023 con fecha 16 de septiembre de 2023,
en sesión extraordinaria de Consejo Transitorio se designó al Órgano Colegiado
Superior del Instituto Superior Tecnológico Yaruqui.
Artículo 1.- Aprobar del Reglamento de la Unidad de Relaciones Públicas y
Comunicación
"""
token_classifier(texto)
```

SALIDA ESPERADA: 
```
[{'entity_group': 'FECHA',
  'score': np.float32(0.9520282),
  'word': '16 de septiembre de 2023',
  'start': 58,
  'end': 82},
 {'entity_group': 'CARGO',
  'score': np.float32(0.63179815),
  'word': 'órgano colegiado superior del instituto superior tecnológico yaruqui',
  'start': 178,
  'end': 246},
 {'entity_group': 'FECHA',
  'score': np.float32(0.9284722),
  'word': '16 de septiembre de 2023',
  'start': 307,
  'end': 331},
 {'entity_group': 'CARGO',
  'score': np.float32(0.5780228),
  'word': 'órgano colegiado superior del instituto superior tecnológico yaruqui',
  'start': 395,
  'end': 463},
 {'entity_group': 'REGLAMENTO',
  'score': np.float32(0.6649067),
  'word': 'reglamento de la unidad de relaciones públicas y comunicación',
  'start': 490,
  'end': 551}]
```