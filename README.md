# DataMiningProj1_2020
------

## How to prepare

1. Install anaconda
2. Install jupytext ( https://anaconda.org/conda-forge/jupytext )
3. Open terminal and run `anaconda-navigator`
4. Launch Jupyter
5. Browse and open .py file. You will see that notebook file will be generated automatically.
    - Go to Jupytext option and tick "Autosave notebook"
6. When you press save, .py and .notebook file are updated automatically.

------

## Σημαντικές παρατηρήσεις

- Κάνουμε add συγκεκριμένα αρχεία και όχι `add *` γιατί αφενός δεν είναι σωστή τεχνική και αφετέρου μπορεί να δημιουργούνται κάποια checkpoint αρχεία του jupyter notebook που δε τα χρειαζόμαστε.
- Έχουμε το φάκελο με τα data από την e-class τοπικά στον υπολογιστή μας και δε τα ανεβάζουμε στο git διότι είναι μεγάλα σε μέγεθος.

------

## TO-DOS

1. ~~Ανοίγουμε 1-1 τα csv αρχεία από τα data και βλέπουμε τα ονόματα των columns που έχει το καθένα. Τα καταγράφουμε κάπου συγκετρωτικά, για να ξέρουμε τι column έχει το κάθε αρχείο ώστε να ξέρουμε τι θα χρειαστούμε. Αυτό μπορεί να γίνει είτε πρόχειρα ανοίγοντας τα αρχεία με κάποιο editor ή excel/libre calulator ή ακόμη καλύτερα με κάποιο "πρόχειρο" πρόγραμμα με python με χρήση pandas κλπ~~
    - ~~Το ιδανικό θα ήταν να μπορέσει να φανεί αυτή η προεπεξεργασία και η συγκέντρωση των στηλών στο τελικό notebook.~~
1. ~~Με βάση την προηγούμενη συσχέτιση φτιάχνουμε το ενιαίο αρχείο που ζητείται στην εκφώνηση και ελέγχουμε ότι είναι οκ. Όλα αυτά τα κάνουμε βήμα-βήμα με κατάλληλα comments και περιγραφές στο notebook.~~
1. Βλέπουμε που υπάρχουν missing data στο ενιαίο αρχείο, τα καταγράφουμε (πάλι με αντιστοίχιση κλπ) και αποφασίζουμε πως θα τα συμπληρώσουμε.
1. Αφού αποφασίσουμε πως τα συμπληρώνουμε, τα συμπληρώνουμε. Πάντα βήμα-βήμα στον κώδικα και με κατάλληλα comments και περιγραφές.
1. Βλέπουμε ποια ερωτήματα μπορούν να γίνουν ανεξάρτητα (δηλαδή να δουλεύουμε και οι 2 παράλληλα χωρίς να εξαρτιόμαστε ο ένας από τον άλλον και τα καταγράφουμε).
1. Σημαντικές παρατηρήσεις με βάση e-class:
    - Να δούμε error/warning με mixed dtypes - προς το παρόν έχω βάλει στο pandas read να τα διαβάζει ως unicode
    - Να μην υπάρχουν δεδομένα που επαναλαμβάνονται (check σχετική ερώτηση στο e-class)
    - Μας καλύπτει η εξτρα στήλη του Μήνα ή πρέπει να ελέγξουμε τις ημερομηνίες (δηλαδή αντιστοιχούν οι ημερομηνίες στην έξτρα στήλη του μήνα);;
1. Ερωτήματα που πρέπει να γίνουν:
    1. ~~Ο πιο συχνός τύπος room~~
    1. Πορεία τιμών για το διάστημα των 3 μηνών.
    1. ~~5 πρώτες γειτονιές με τις περισσότερες κριτικές~~ (__Τι κάνουμε με τις NaN γειτονιές;;;__)
    1. Γειτονιά με τα περισσότερα ακίνητα (__Τι κάνουμε με τις NaN γειτονιές;;;__)
    1.  ~~Καταχωρίσεις ανά γειτονιά και ανά μήνα ~~ (__Τι κάνουμε με τις NaN γειτονιές;;;__) (__Μήπως πρέπει να κάνω count distinct Ids;;;__)
    (__Εννοεί ξεχωριστά ανά γειτονιά και ανά μήνα ή και τα 2 μαζί;;;__)
    1. Ιστόγραμμα neighborhood
    1. Πιο συχνό room_type σε κάθε γειτονιά.
    1. Ο πιο ακριβός τύπος δωματίου
    1. Folium Map κλπ
    1. Wordclouds
    1. Extra 2 questions
1. Σκεφτόμαστε για το ερώτημα 1.12.
1. Εφόσον ξεμπερδέψουμε με τα ερωτήματα 1 (ή και πριν τελειώσουμε) μπορούμε να αναλύσουμε και να σχεδιάσουμε το ερώτημα 3.
    - Μελετάμε τι βήματα χρειάζονται, πως μπορεί να χωριστεί κλπ