"""Evaluation dataset definition, generator, and frozen loader for Open-XTTS-Trainer.

Defines a fixed evaluation set versioned in the repo:
- English: 50 held-out, phonetically varied sentences (sibilants, numbers, long sentences, questions, Harvard).
- Target languages (es, fr, de, hi): 30 authentic native sentences each.
- Reference voices: 5 distinct speakers (female/male, diverse accents).
"""

import json
import hashlib
import os
from typing import Dict, Any, List


# 50 Phonetically varied English sentences
ENGLISH_EVAL_SENTENCES: List[Dict[str, Any]] = [
    # Harvard / Phonetic Balance (1-10)
    {"id": "en_01", "category": "harvard", "text": "The birch canoe slid on the smooth planks."},
    {"id": "en_02", "category": "harvard", "text": "Glue the sheet to the dark blue background."},
    {"id": "en_03", "category": "harvard", "text": "It is easy to tell the depth of a well."},
    {"id": "en_04", "category": "harvard", "text": "These days a chicken leg is a rare dish."},
    {"id": "en_05", "category": "harvard", "text": "Rice is often served in round bowls."},
    {"id": "en_06", "category": "harvard", "text": "The juice of lemons makes fine punch."},
    {"id": "en_07", "category": "harvard", "text": "A box of red pens was on the table."},
    {"id": "en_08", "category": "harvard", "text": "The salt breeze came through from the open sea."},
    {"id": "en_09", "category": "harvard", "text": "The wagon moved on well oiled wheels."},
    {"id": "en_10", "category": "harvard", "text": "March the soldiers back across the wide river."},

    # Sibilants & High-Frequency Stress (11-20)
    {"id": "en_11", "category": "sibilant", "text": "She sells sea shells by the seashore with fresh sunshine."},
    {"id": "en_12", "category": "sibilant", "text": "The six sick sheiks' sixth sheep is sick."},
    {"id": "en_13", "category": "sibilant", "text": "Sally swiftly swept seven slippery steps outside the museum."},
    {"id": "en_14", "category": "sibilant", "text": "Crisp sparkling cider was served in silver glasses."},
    {"id": "en_15", "category": "sibilant", "text": "Silence settled softly over the distant misty seashore."},
    {"id": "en_16", "category": "sibilant", "text": "A sudden splash startled the sleeping black swans."},
    {"id": "en_17", "category": "sibilant", "text": "The sharp breeze whistled swiftly through the pine trees."},
    {"id": "en_18", "category": "sibilant", "text": "Special scientists studied celestial satellites in space."},
    {"id": "en_19", "category": "sibilant", "text": "Six silly sisters stood beside the singing stream."},
    {"id": "en_20", "category": "sibilant", "text": "The silver whistle sounded sharply across the stadium."},

    # Numbers, Dates & Currencies (21-30)
    {"id": "en_21", "category": "numbers", "text": "The flight departs at 07:30 from gate 42 with 150 passengers."},
    {"id": "en_22", "category": "numbers", "text": "In 1984, the population reached 4,800,000 across 350 districts."},
    {"id": "en_23", "category": "numbers", "text": "The ticket costs $25.50 for adults and $12.00 for children."},
    {"id": "en_24", "category": "numbers", "text": "Room 204 is located on the second floor of building 5."},
    {"id": "en_25", "category": "numbers", "text": "The temperature dropped to minus 15 degrees at midnight."},
    {"id": "en_26", "category": "numbers", "text": "He scored 98 points out of 100 on the final chemistry exam."},
    {"id": "en_27", "category": "numbers", "text": "The company reported a 14.5 percent revenue increase in quarter 3."},
    {"id": "en_28", "category": "numbers", "text": "The distance between the two cities is roughly 320 kilometers."},
    {"id": "en_29", "category": "numbers", "text": "The telephone number to call is 555-0199 extension 408."},
    {"id": "en_30", "category": "numbers", "text": "We need 3 boxes of nails, 12 wooden boards, and 50 screws."},

    # Long & Complex Sentences (31-40)
    {"id": "en_31", "category": "long_complex", "text": "Artificial intelligence is rapidly transforming how we communicate, work, and build open technology across distributed computing clusters."},
    {"id": "en_32", "category": "long_complex", "text": "The comprehensive preservation of historical architectural monuments requires sustained cooperation among civil engineers, historians, and urban planners."},
    {"id": "en_33", "category": "long_complex", "text": "Recent astronomical observations suggest that water ice may be widely distributed across permanently shadowed craters near the lunar south pole."},
    {"id": "en_34", "category": "long_complex", "text": "Technological advancements in renewable energy storage have dramatically lowered the total cost of operating municipal microgrids."},
    {"id": "en_35", "category": "long_complex", "text": "Modern neural text to speech synthesizers combine discrete acoustic tokenizers with deep autoregressive language models for expressive prosody."},
    {"id": "en_36", "category": "long_complex", "text": "International meteorological organizations coordinate satellite observations to deliver real-time warnings before tropical storms make landfall."},
    {"id": "en_37", "category": "long_complex", "text": "Researchers at the university published a thorough investigation into the long-term ecological benefits of temperate wetland restoration."},
    {"id": "en_38", "category": "long_complex", "text": "Efficient zero-shot voice cloning enables instant adaptation to target speaker characteristics using only a few seconds of conditioning audio."},
    {"id": "en_39", "category": "long_complex", "text": "The development of reliable quantum error correction codes remains one of the central challenges in contemporary physics and computer engineering."},
    {"id": "en_40", "category": "long_complex", "text": "Sustainable agricultural practices must balance groundwater conservation with crop yield optimization under changing climatic conditions."},

    # Questions, Exclamations & Intonation (41-50)
    {"id": "en_41", "category": "prosody_questions", "text": "Could you please explain how the acoustic feature extraction works in real time?"},
    {"id": "en_42", "category": "prosody_questions", "text": "Where did you leave the keys to the sound recording studio yesterday afternoon?"},
    {"id": "en_43", "category": "prosody_questions", "text": "What an extraordinary performance by the symphony orchestra tonight!"},
    {"id": "en_44", "category": "prosody_questions", "text": "Have you considered testing the speech synthesis pipeline on multilingual data?"},
    {"id": "en_45", "category": "prosody_questions", "text": "Why did the system administrator choose to reboot the primary server during peak hours?"},
    {"id": "en_46", "category": "prosody_questions", "text": "Listen carefully to the high-frequency harmonics produced by the acoustic violin."},
    {"id": "en_47", "category": "prosody_questions", "text": "Would you prefer to travel by express train or take the scenic coastal highway?"},
    {"id": "en_48", "category": "prosody_questions", "text": "How many neural network layers are used in the generative adversarial decoder?"},
    {"id": "en_49", "category": "prosody_questions", "text": "Please remember to verify the integrity checksum before installing the new software package."},
    {"id": "en_50", "category": "prosody_questions", "text": "Are the synthesized waveforms indistinguishable from authentic human voice recordings?"},
    # Short utterances (1-3 words): the community's most-reported failure is hallucination on short text
    # (docs/recipe_design.md §3, ablation A1). Common homophones (no/know) avoided so errors reflect synthesis.
    {"id": "en_51", "category": "short", "text": "Yes."},
    {"id": "en_52", "category": "short", "text": "Thanks!"},
    {"id": "en_53", "category": "short", "text": "Seven."},
    {"id": "en_54", "category": "short", "text": "Why?"},
    {"id": "en_55", "category": "short", "text": "Hello there."},
    {"id": "en_56", "category": "short", "text": "Good morning."},
    {"id": "en_57", "category": "short", "text": "See you soon."},
    {"id": "en_58", "category": "short", "text": "Absolutely not."},
    {"id": "en_59", "category": "short", "text": "Wait for me."},
    {"id": "en_60", "category": "short", "text": "Call me tomorrow."},
]


# 30 Spanish Native Sentences (es)
SPANISH_EVAL_SENTENCES: List[Dict[str, Any]] = [
    {"id": "es_01", "text": "El sol brillaba con fuerza sobre las montañas nevadas de la cordillera."},
    {"id": "es_02", "text": "La biblioteca municipal abre sus puertas de lunes a viernes a las nueve de la mañana."},
    {"id": "es_03", "text": "Los investigadores encontraron nuevas evidencias arqueológicas en la cueva submarina."},
    {"id": "es_04", "text": "¿Podrías indicarme cuál es el camino más rápido para llegar a la estación central?"},
    {"id": "es_05", "text": "El aroma del café recién tostado llenaba cada rincón de la pequeña pastelería."},
    {"id": "es_06", "text": "En el año 1992 se celebraron los Juegos Olímpicos en la hermosa ciudad de Barcelona."},
    {"id": "es_07", "text": "La orquesta filarmónica interpretó una sinfonía clásica ante más de mil quinientos espectadores."},
    {"id": "es_08", "text": "Todos los pasajeros deben permanecer sentados con el cinturón de seguridad abrochado."},
    {"id": "es_09", "text": "El desarrollo de modelos neuronales abiertos promueve la democratización de la tecnología."},
    {"id": "es_10", "text": "Las olas del mar chocaban suavemente contra las rocas del acantilado al atardecer."},
    {"id": "es_11", "text": "El precio del billete de autobús aumentó un cinco por ciento a principios de este mes."},
    {"id": "es_12", "text": "Mañana por la tarde visitaremos el museo nacional de arte contemporáneo con los estudiantes."},
    {"id": "es_13", "text": "La brisa fresca traía consigo el perfume silvestre de los campos de lavanda."},
    {"id": "es_14", "text": "¿Cuántas personas asistieron a la conferencia sobre inteligencia artificial y ética?"},
    {"id": "es_15", "text": "El médico recomendó beber al menos dos litros de agua y descansar adecuadamente."},
    {"id": "es_16", "text": "Un grupo de biólogos marinos descubrió una nueva especie de coral en el arrecife."},
    {"id": "es_17", "text": "Los niños jugaban alegremente en el parque mientras caían las hojas doradas del otoño."},
    {"id": "es_18", "text": "La tecnología de clonación vocal permite generar voces realistas a partir de grabaciones cortas."},
    {"id": "es_19", "text": "El tren de alta velocidad conecta las dos capitales en menos de dos horas y media."},
    {"id": "es_20", "text": "Por favor, apague los dispositivos electrónicos antes del despegue y aterrizaje del avión."},
    {"id": "es_21", "text": "La cosecha de olivas de este año fue catalogada como una de las mejores de la década."},
    {"id": "es_22", "text": "¿Cuál es tu opinión acerca del impacto de la automatización en el empleo moderno?"},
    {"id": "es_23", "text": "El río desbordó su cauce habitual tras tres días consecutivos de lluvias torrenciales."},
    {"id": "es_24", "text": "Cincuenta voluntarios limpiaron la playa recogiendo más de doscientas bolsas de residuos."},
    {"id": "es_25", "text": "El silencio absoluto de la noche fue interrumpido por el canto lejano de un búho."},
    {"id": "es_26", "text": "La Universidad anunció la creación de una nueva beca para proyectos de investigación científica."},
    {"id": "es_27", "text": "Un cielo despejado y una luna llena iluminaron el sendero a través del espeso bosque."},
    {"id": "es_28", "text": "La empresa exporta productos agrícolas de alta calidad a más de veinticinco países."},
    {"id": "es_29", "text": "¿Has terminado de redactar el informe técnico que solicitó el comité directivo?"},
    {"id": "es_30", "text": "La música tradicional española refleja la rica diversidad cultural de cada región peninsular."},
]


# 30 French Native Sentences (fr)
FRENCH_EVAL_SENTENCES: List[Dict[str, Any]] = [
    {"id": "fr_01", "text": "Le soleil matinal illuminait doucement les toits dorés de la capitale."},
    {"id": "fr_02", "text": "La bibliothèque universitaire propose plus de trois cent mille ouvrages aux étudiants."},
    {"id": "fr_03", "text": "Pourriez-vous m'indiquer le chemin le plus court pour rejoindre le musée du Louvre ?"},
    {"id": "fr_04", "text": "Le parfum agréable du pain chaud sortant du four embaumait toute la petite ruelle."},
    {"id": "fr_05", "text": "Les chercheurs ont publié une étude comparative sur la synthèse vocale neuronale."},
    {"id": "fr_06", "text": "Le train à grande vitesse arrivera en gare avec cinq minutes d'avance sur l'horaire prévu."},
    {"id": "fr_07", "text": "En hiver, la neige recouvre les cimes majestueuses des Alpes d'un épais manteau blanc."},
    {"id": "fr_08", "text": "Veuillez attacher votre ceinture de sécurité et éteindre tous vos appareils portables."},
    {"id": "fr_09", "text": "Le festival international de cinéma accueille cette année des réalisateurs du monde entier."},
    {"id": "fr_10", "text": "Le vent d'automne faisait tournoyer doucement les feuilles rousses le long des boulevards."},
    {"id": "fr_11", "text": "Combien de temps faut-il pour apprendre à parler couramment une langue étrangère ?"},
    {"id": "fr_12", "text": "La réunion scientifique annuelle débutera demain matin à huit heures précises dans l'amphithéâtre."},
    {"id": "fr_13", "text": "Une promenade le long des quais de la Seine offre une vue exceptionnelle sur les monuments."},
    {"id": "fr_14", "text": "Les progrès technologiques récents ouvrent des perspectives inédites dans le domaine médical."},
    {"id": "fr_15", "text": "Le chef cuisinier a préparé une soupe de légumes de saison accompagnée de fines herbes."},
    {"id": "fr_16", "text": "Les oiseaux migrateurs traversent le ciel en formation serrée vers le sud chaleureux."},
    {"id": "fr_17", "text": "Le musée sera exceptionnellement fermé au public le premier dimanche du mois prochain."},
    {"id": "fr_18", "text": "Avez-vous pensé à vérifier les données avant de lancer le calcul d'optimisation ?"},
    {"id": "fr_19", "text": "La conférence sur l'écologie a réuni plus de deux mille participants venus de quarante pays."},
    {"id": "fr_20", "text": "Un silence paisible régnait dans la cathédrale éclairée par les reflets des vitraux."},
    {"id": "fr_21", "text": "La production d'énergie solaire a augmenté de douze pour cent au cours de l'année écoulée."},
    {"id": "fr_22", "text": "Le pianiste virtuose a reçu une ovation debout prolongée après son interprétation remarquable."},
    {"id": "fr_23", "text": "Les enfants riaient aux éclats en découvrant les animaux marins dans le grand aquarium."},
    {"id": "fr_24", "text": "Il est essentiel de maintenir un équilibre sain entre vie professionnelle et détente personnelle."},
    {"id": "fr_25", "text": "La rivière serpente calmement à travers les vallées verdoyantes et les villages pittoresques."},
    {"id": "fr_26", "text": "Quelles sont les principales différences entre ces deux architectures d'apprentissage profond ?"},
    {"id": "fr_27", "text": "Le marché hebdomadaire regorge de fruits frais, de fromages artisanaux et de fleurs colorées."},
    {"id": "fr_28", "text": "La température extérieure est descendue sous zéro dès la tombée de la nuit étoilée."},
    {"id": "fr_29", "text": "Les astronomes ont détecté un nouveau signal radio provenant d'une galaxie très lointaine."},
    {"id": "fr_30", "text": "Une mélodie douce résonnait à travers les murs anciens de la vieille demeure familiale."},
]


# 30 German Native Sentences (de)
GERMAN_EVAL_SENTENCES: List[Dict[str, Any]] = [
    {"id": "de_01", "text": "Die Morgensonne schien hell über die schneebedeckten Gipfel der bayerischen Alpen."},
    {"id": "de_02", "text": "Die Universitätsbibliothek öffnet täglich um acht Uhr morgens für alle Studierenden."},
    {"id": "de_03", "text": "Könnten Sie mir bitte den kürzesten Weg zum historischen Marktplatz beschreiben?"},
    {"id": "de_04", "text": "Der Duft von frisch gebackenem Brot erfüllte die gemütliche Bäckerei in der Altstadt."},
    {"id": "de_05", "text": "Wissenschaftler haben neue Methoden zur Optimierung neuronaler Sprachmodelle entwickelt."},
    {"id": "de_06", "text": "Der ICE-Zug erreicht den Berliner Hauptbahnhof pünktlich um vierzehn Uhr dreißig."},
    {"id": "de_07", "text": "Im Herbst färben sich die Blätter der alten Buchenbäume in leuchtendem Gold und Rot."},
    {"id": "de_08", "text": "Bitte schnallen Sie sich an und beachten Sie die Hinweise des Kabinenpersonals."},
    {"id": "de_09", "text": "Das renommierte Orchester spielte eine Symphonie von Beethoven vor ausverkauftem Haus."},
    {"id": "de_10", "text": "Ein kalter Wind wehte über die weiten Felder und kündigte den nahenden Winter an."},
    {"id": "de_11", "text": "Wie viele Teilnehmer haben sich für den Workshop über maschinelles Lernen angemeldet?"},
    {"id": "de_12", "text": "Die Bundesregierung beschloss neue Fördermaßnahmen für erneuerbare Energien und Klimaschutz."},
    {"id": "de_13", "text": "Ein Spaziergang am Ufer des Sees bietet Ruhe und eine wunderschöne Aussicht ins Tal."},
    {"id": "de_14", "text": "Der Arzt empfahl ausreichend Schlaf und regelmäßige Bewegung an der frischen Luft."},
    {"id": "de_15", "text": "Die Firma investierte fünf Millionen Euro in modernste Automatisierungstechnik."},
    {"id": "de_16", "text": "In den alten Wäldern des Schwarzwaldes findet man seltene Pflanzen und Vogelarten."},
    {"id": "de_17", "text": "Haben Sie alle Dokumente sorgfältig geprüft, bevor Sie den Vertrag unterschrieben haben?"},
    {"id": "de_18", "text": "Die Straßenbahnlinie zwölf verbindet das Universitätsklinikum mit dem Stadtzentrum."},
    {"id": "de_19", "text": "Moderne Sprachsynthese erfordert eine präzise Modellierung von Tonhöhe und Rhythmus."},
    {"id": "de_20", "text": "Die Kinder bauten im verschneiten Garten einen großen Schneemann mit Möhrennase."},
    {"id": "de_21", "text": "Der jährliche Kongress für Informatik findet dieses Jahr im Kongresszentrum statt."},
    {"id": "de_22", "text": "Die Ernte von Äpfeln und Birnen fiel in diesem Spätsommer besonders ertragreich aus."},
    {"id": "de_23", "text": "Welche Vorteile bieten Open-Source-Frameworks für die wissenschaftliche Forschung?"},
    {"id": "de_24", "text": "Der Rhein fließt majestätisch an mittelalterlichen Burgen und steilen Weinbergen vorbei."},
    {"id": "de_25", "text": "Auf dem Wochenmarkt bieten regionale Bauern frisches Gemüse, Käse und Honig an."},
    {"id": "de_26", "text": "Ein dichter Nebel lag über den Wiesen und löste sich erst gegen Mittag langsam auf."},
    {"id": "de_27", "text": "Das Konzert beginnt um neunzehn Uhr mit einer kurzen Einführung des Dirigenten."},
    {"id": "de_28", "text": "Die Luftqualität in den Großstädten hat sich dank moderner Filtertechnik verbessert."},
    {"id": "de_29", "text": "Können Sie die Ergebnisse der statistischen Analyse bis morgen zusammenfassen?"},
    {"id": "de_30", "text": "Ein gutes Buch und eine Tasse heißer Tee sind an regnerischen Abenden unersetzlich."},
]


# 30 Hindi Native Sentences (hi)
HINDI_EVAL_SENTENCES: List[Dict[str, Any]] = [
    {"id": "hi_01", "text": "सुबह का सूरज हिमालय की बर्फीली चोटियों पर सुनहरी किरणें बिखेर रहा था।"},
    {"id": "hi_02", "text": "विश्वविद्यालय का केंद्रीय पुस्तकालय सभी छात्रों के लिए सुबह नौ बजे खुलता है।"},
    {"id": "hi_03", "text": "क्या आप मुझे मुख्य रेलवे स्टेशन तक जाने का सबसे छोटा रास्ता बता सकते हैं?"},
    {"id": "hi_04", "text": "गर्म चाय और ताजी जलेबी की खुशबू से पूरी छोटी दुकान महक रही थी।"},
    {"id": "hi_05", "text": "भारतीय वैज्ञानिकों ने अंतरिक्ष अनुसंधान के क्षेत्र में कई ऐतिहासिक सफलताएं हासिल की हैं।"},
    {"id": "hi_06", "text": "तेज गति वाली एक्सप्रेस ट्रेन अपने निर्धारित समय से दस मिनट पहले पहुंच गई।"},
    {"id": "hi_07", "text": "कृपया अपनी सीट बेल्ट बांध लें और विमान में सुरक्षा नियमों का पालन करें।"},
    {"id": "hi_08", "text": "वार्षिक संगीत समारोह में देश के प्रसिद्ध शास्त्रीय गायकों ने अपनी प्रस्तुति दी।"},
    {"id": "hi_09", "text": "नदी के किनारे ठंडी हवा बह रही थी और शाम का वातावरण बेहद शांत और सुंदर था।"},
    {"id": "hi_10", "text": "कृत्रिम बुद्धिमत्ता आधुनिक शिक्षा और स्वास्थ्य सेवाओं में क्रांतिकारी बदलाव ला रही है।"},
    {"id": "hi_11", "text": "इस वर्ष अच्छी मानसूनी बारिश के कारण खरीफ की फसल बहुत अच्छी हुई है।"},
    {"id": "hi_12", "text": "क्या आपने सम्मेलन में प्रस्तुत किए जाने वाले शोध पत्र की समीक्षा पूरी कर ली है?"},
    {"id": "hi_13", "text": "बच्चों ने विद्यालय के वार्षिक खेल दिवस में बड़े उत्साह और आनंद के साथ भाग लिया।"},
    {"id": "hi_14", "text": "स्वस्थ जीवन के लिए नियमित योग, संतुलित आहार और पर्याप्त नींद अत्यंत आवश्यक हैं।"},
    {"id": "hi_15", "text": "नई दिल्ली और आगरा के बीच एक्सप्रेसवे पर यात्रा का समय काफी कम हो गया है।"},
    {"id": "hi_16", "text": "पर्यावरण संरक्षण के लिए प्लास्टिक के उपयोग को रोकना हम सबकी साझा जिम्मेदारी है।"},
    {"id": "hi_17", "text": "बाजार में ताजे फल, हरी सब्जियां और पारंपरिक हस्तशिल्प की भरपूर किस्में उपलब्ध थीं।"},
    {"id": "hi_18", "text": "ध्वनि संश्लेषण की नई तकनीकें किसी भी आवाज की स्पष्टता को बेहतर बनाती हैं।"},
    {"id": "hi_19", "text": "रात के समय आकाश में चमकते अनगिनत सितारे प्रकृति की सुंदरता को दर्शाते हैं।"},
    {"id": "hi_20", "text": "क्या आप बता सकते हैं कि इस कंप्यूटर प्रणाली की मुख्य विशेषताएं क्या हैं?"},
    {"id": "hi_21", "text": "प्राचीन काल से ही भारतीय साहित्य और दर्शन विश्व भर में प्रेरणा का स्रोत रहे हैं।"},
    {"id": "hi_22", "text": "ग्राम पंचायत ने स्वच्छ पेयजल और सौर ऊर्जा के लिए नई परियोजनाएं शुरू की हैं।"},
    {"id": "hi_23", "text": "संग्रहालय में रखी गई ऐतिहासिक वस्तुएं भारत की समृद्ध विरासत का प्रमाण हैं।"},
    {"id": "hi_24", "text": "दूरदर्शन पर प्रसारित समाचार बुलेटिन ने देश-विदेश की प्रमुख घटनाओं की जानकारी दी।"},
    {"id": "hi_25", "text": "वसंत ऋतु के आगमन के साथ ही बगीचों में रंग-बिरंगे फूल खिलने लगे हैं।"},
    {"id": "hi_26", "text": "वैज्ञानिकों ने नवीकरणीय ऊर्जा स्रोतों की कार्यक्षमता बढ़ाने के लिए नए प्रयोग किए।"},
    {"id": "hi_27", "text": "इस परियोजना को समय पर पूरा करने के लिए सभी इंजीनियरों ने दिन-रात परिश्रम किया।"},
    {"id": "hi_28", "text": "सड़क पर सुरक्षित यात्रा के लिए यातायात नियमों का कड़ाई से पालन करना अनिवार्य है।"},
    {"id": "hi_29", "text": "क्या आप अगले सप्ताह होने वाले अंतरराष्ट्रीय संगोष्ठी में भाग लेने आ रहे हैं?"},
    {"id": "hi_30", "text": "गुरु शिष्य परंपरा भारतीय संस्कृति और संगीत शिक्षा की एक अमूल्य पहचान है।"},
]


# 5 held-out reference voices from CMU ARCTIC (16 kHz studio recordings; license file ships in each
# tarball and is copied next to the built clips). Each reference clip is the listed utterances, in
# order, with edge silence trimmed and 0.25 s gaps, stopping once it reaches 6 s and capped at 10 s.
# Built and hash-locked by scripts/fetch_reference_voices.py (eval/reference_voices.py).
CMU_ARCTIC_URL = "http://festvox.org/cmu_arctic/packed/cmu_us_{speaker}_arctic.tar.bz2"
# The packed tarballs omit the license; each speaker's COPYING is published with the unpacked database.
# Terms: free for any purpose (commercial or otherwise) if the notice is kept, modifications are clearly
# marked, and the original authors' names are not deleted.
CMU_ARCTIC_LICENSE_URL = "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{speaker}_arctic/COPYING"
REFERENCE_UTTERANCES = ["arctic_a0001", "arctic_a0002", "arctic_a0003", "arctic_a0004", "arctic_a0005", "arctic_a0006"]

REFERENCE_VOICES: List[Dict[str, Any]] = [
    {
        "speaker_id": "arctic_slt",
        "corpus": "CMU ARCTIC",
        "corpus_speaker": "slt",
        "gender": "female",
        "language": "en",
        "accent": "US English",
        "utterances": REFERENCE_UTTERANCES,
    },
    {
        "speaker_id": "arctic_clb",
        "corpus": "CMU ARCTIC",
        "corpus_speaker": "clb",
        "gender": "female",
        "language": "en",
        "accent": "US English",
        "utterances": REFERENCE_UTTERANCES,
    },
    {
        "speaker_id": "arctic_bdl",
        "corpus": "CMU ARCTIC",
        "corpus_speaker": "bdl",
        "gender": "male",
        "language": "en",
        "accent": "US English",
        "utterances": REFERENCE_UTTERANCES,
    },
    {
        "speaker_id": "arctic_awb",
        "corpus": "CMU ARCTIC",
        "corpus_speaker": "awb",
        "gender": "male",
        "language": "en",
        "accent": "Scottish English",
        "utterances": REFERENCE_UTTERANCES,
    },
    {
        "speaker_id": "arctic_ksp",
        "corpus": "CMU ARCTIC",
        "corpus_speaker": "ksp",
        "gender": "male",
        "language": "en",
        "accent": "Indian English",
        "utterances": REFERENCE_UTTERANCES,
    },
]


def build_evaluation_dataset() -> Dict[str, Any]:
    """Build the comprehensive, frozen evaluation dataset dictionary."""
    data = {
        "version": "1.2.0",
        "created_date": "2026-10-24",
        "description": "Frozen multi-speaker, multilingual ground-truth evaluation set for Open-XTTS-Trainer. v1.1.0: placeholder voices replaced with CMU ARCTIC speakers. v1.2.0: short-utterance category (en_51-en_60).",
        "counts": {
            "english": len(ENGLISH_EVAL_SENTENCES),
            "spanish": len(SPANISH_EVAL_SENTENCES),
            "french": len(FRENCH_EVAL_SENTENCES),
            "german": len(GERMAN_EVAL_SENTENCES),
            "hindi": len(HINDI_EVAL_SENTENCES),
            "total_sentences": (
                len(ENGLISH_EVAL_SENTENCES)
                + len(SPANISH_EVAL_SENTENCES)
                + len(FRENCH_EVAL_SENTENCES)
                + len(GERMAN_EVAL_SENTENCES)
                + len(HINDI_EVAL_SENTENCES)
            ),
            "reference_voices": len(REFERENCE_VOICES),
        },
        "english_sentences": ENGLISH_EVAL_SENTENCES,
        "target_languages": {
            "es": SPANISH_EVAL_SENTENCES,
            "fr": FRENCH_EVAL_SENTENCES,
            "de": GERMAN_EVAL_SENTENCES,
            "hi": HINDI_EVAL_SENTENCES,
        },
        "reference_voices": REFERENCE_VOICES,
    }
    return data


# Absolute, so a script or test run from another directory reads the frozen set instead of missing it
DEFAULT_EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "eval", "dataset", "eval_set.json")


def eval_set_content_hash(data: Dict[str, Any]) -> str:
    """SHA-256 of the eval set's content: its JSON without the "sha256" field, as save_frozen_eval_set writes it."""
    content = {k: v for k, v in data.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(content, indent=2, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_frozen_eval_set(output_path: str) -> str:
    """Save the evaluation set and return its SHA-256 hash.

    output_path has no default: writing to DEFAULT_EVAL_SET_PATH replaces the frozen set every baseline is
    measured against, so it has to be named on purpose.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dataset = build_evaluation_dataset()
    sha256 = eval_set_content_hash(dataset)
    dataset["sha256"] = sha256

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return sha256


def load_eval_set(path: str = DEFAULT_EVAL_SET_PATH) -> Dict[str, Any]:
    """Load the frozen evaluation set from disk with validation.

    A missing file is an error. Regenerating it here (as this used to) would silently measure against a new
    file whenever a path was wrong, e.g. a relative default resolved from another working directory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No frozen eval set at {path}; it is committed at {DEFAULT_EVAL_SET_PATH}. "
                                f"Freezing a new one is deliberate: eval.dataset.save_frozen_eval_set(path)")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Manifests record data["sha256"], so it must describe this content: an edited set can't keep its old hash
    actual = eval_set_content_hash(data)
    if data.get("sha256") != actual:
        raise ValueError(f"{path}: stored sha256 {data.get('sha256')} doesn't match its content ({actual}); "
                         f"the frozen eval set was edited. Restore it, or freeze a new one deliberately")

    # Basic structural validation
    assert "english_sentences" in data, "Missing 'english_sentences' in eval set"
    assert len(data["english_sentences"]) >= 50, f"Expected >= 50 English sentences, got {len(data['english_sentences'])}"
    assert "target_languages" in data, "Missing 'target_languages' in eval set"
    for lang, sents in data["target_languages"].items():
        assert len(sents) >= 30, f"Expected >= 30 sentences for lang {lang}, got {len(sents)}"
    assert "reference_voices" in data, "Missing 'reference_voices' in eval set"
    assert len(data["reference_voices"]) >= 5, f"Expected >= 5 reference voices, got {len(data['reference_voices'])}"

    return data
