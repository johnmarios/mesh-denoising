# Question 6 — holes / MeshLab

Το Q6 παραμένει ξεχωριστό από το Python denoising pipeline επειδή στην τρέχουσα εργασία η αποκατάσταση οπών γίνεται με MeshLab / remeshing εργαλεία.

Όταν φτάσουμε εδώ θα καταγράψουμε:

1. πώς δημιουργείται / εντοπίζεται η οπή,
2. ποια MeshLab μέθοδος εφαρμόζεται,
3. τι αλλάζει στην τοπολογία,
4. γιατί correspondence-based RMSE/MSAE δεν είναι πάντα έγκυρα μετά το remeshing,
5. γιατί χρειαζόμαστε surface-based / symmetric evaluation.

Δεν βάζουμε placeholder Python algorithm μόνο και μόνο για να υπάρχει αρχείο.
