# Projet 7 parcours Data Scientist Openclassrooms
## Implémentez un modèle de scoring
### Partie API 

Déployer un api sur Azur qui renvoit la probabilité de défault de crédit d'un client
On met en input l'ID client et on reçoit en output la probabilité de défault et la matrice des shap values du modèle

Les fichiers : 
vs_code_api : l'api fastAPI
requirements : versionning des packages python
start.sh : script le lancement de l'application dans azur
model.pkl : modèle version pickle 
test_df.pkl : dataframe contenant les données clients
test_samp.pkl : échantillon du dataframe test_df.pkl

#### Test unitaires 
dossier tests/test
déploiement continu avec Github Action
Vérifie que les données de l'api sont au bon format, JSON, dictionnaire, Array pour les shap values.
Vérifie que la probabilité soit bien comprise entre 0 et 1
