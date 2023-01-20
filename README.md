# Neural network from scratch in Python

### Utfört av: Jacob Nilsson & Jacob Lundkvist
### Handledare: Erik  Pihl
****

## Introduktion
Detta program gick ut på att från grunden konstruera ett neuralt nätverk. Inga externa biblotek får användas för att konstruera nätveket. Målet med nätveket var att kunna predikera utdatan från fyra tryckknappar kopplat till ett raspberry pi. Enligt uppgiften så skall en lysdiod lysa ifall ett ojämnt antal knappar trycks ner samtidigt och vara släckt när ett jämnt antal knappar trycks ner.

***

## Resultat 
Resultatet blev som förväntat då vi inte söttte på några större beskymmer så flöt allting på bra. Nätverket uppfyller samtliga krav och fungerar felfritt. Träningsdatan läses in från en fil som defineras i main (standard train_data.txt). Sedan så tränas den baserat på de inställningar som skickas med i main funktionen och skriver resultat i filen output.txt. Det bästa inställningarna vi hittat för nätveket är runt 10 000 epoker och med en learn rate på 0.0255 så hittade vi att nätverket har riktigt bra träffsäkerhet men även att det fortfarande går fort att träna.

***
## Diskussion
Projektet var på en bra nivå och var roligt att utföra. Det var riktigt lärorikt och kul att se hur man konstruerar och användningsområden för ett nuralt nätverk. Det fanns mycket bra exempel i både C och C++ från Erik för att kunna ta hjälp av om man körde fast vilket är väldigt bra.

***