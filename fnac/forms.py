from django import forms
class fnacForm(forms.Form):
    
    ClumpThicknessChoices = [(1, "Cells are fully mono-layered"),
                             (2, "Cells are 90% mono-layered"),
                             (3, "Cells are 80% mono-layered"),
                             (4, "Cells are 65% mono-layered"),
                             (5, "Cells are slightly more mono-layered, than multi-layered"),
                             (6, "Cells are slightly more multi-layered, than mono-layered"),
                             (7, "Cells are 35% mono-layered"),
                             (8, "Cells are 20% mono-layered"),
                             (9, "Cells are 10% mono-layered"),
                             (10, "Cells are multi-layered")]
    UniformityofCellSizeChoices = [(1, "Cells are completely uniform"),
                             (2, "Cells are 90% uniform"),
                             (3, "Cells are 80% uniform"),
                             (4, "Cells are 65% uniform"),
                             (5, "Cells are more than 50% uniform"),
                             (6, "Cells are less than 50% uniform"),
                             (7, "Cells are 35% uniform"),
                             (8, "Cells are 20% uniform"),
                             (9, "Cells are 10% uniform"),
                             (10, "Cells are inconsistent with their uniformity")]
    UniformityofCellShapeChoices = [(1, "Cells are completely uniform"),
                             (2, "Cells are 90% uniform"),
                             (3, "Cells are 80% uniform"),
                             (4, "Cells are 65% uniform"),
                             (5, "Cells are more than 50% uniform"),
                             (6, "Cells are less than 50% uniform"),
                             (7, "Cells are 35% uniform"),
                             (8, "Cells are 20% uniform"),
                             (9, "Cells are 10% uniform"),
                             (10, "Cells are inconsistent with their uniformity")]
    MarginalAdhesionChoices = [(1, "Cells completely stick together"),
                             (2, "Cells 90% stick together"),
                             (3, "Cells 80% stick together"),
                             (4, "Cells 70% stick together"),
                             (5, "Cells 60% stick together"),
                             (6, "Cells 50% stick together"),
                             (7, "Cells 40% stick together"),
                             (8, "Cells 30% stick together"),
                             (9, "Cells 20% stick together"),
                             (10, "Cells do not exhibit marginal adhesion")]
    SingleEpithelialCellSizeChoices = [(1, "No cells are significantly enlarged"),
                             (2, "Largest cells appear 20% larger"),
                             (3, "Largest cells appear 30% larger"),
                             (4, "Largest cells appear 40% larger"),
                             (5, "Largest cells appear 50% larger"),
                             (6, "Largest cells appear 60% larger"),
                             (7, "Largest cells appear 70% larger"),
                             (8, "Largest cells appear 80% larger"),
                             (9, "Largest cells appear 90% larger"),
                             (10, "Largest cells appear more than twice their size")]
    BareNucleiChoices = [(1, "Nuclei are completely devoid of cytoplasm"),
                             (2, "20% of nuclei have cytoplasm"),
                             (3, "30% of nuclei have cytoplasm"),
                             (4, "40% of nuclei have cytoplasm"),
                             (5, "50% of nuclei have cytoplasm"),
                             (6, "60% of nuclei have cytoplasm"),
                             (7, "70% of nuclei have cytoplasm"),
                             (8, "80% of nuclei have cytoplasm"),
                             (9, "90% of nuclei have cytoplasm"),
                             (10, "All of nuclei have cytoplasm")]
    BlandChromatinChoices = [(1, "Completely fine-textured chromatin"),
                             (2, "Chromatin is 20% coarse"),
                             (3, "Chromatin is 30% coarse"),
                             (4, "Chromatin is 40% coarse"),
                             (5, "Chromatin is 50% coarse"),
                             (6, "Chromatin is 60% coarse"),
                             (7, "Chromatin is 70% coarse"),
                             (8, "Chromatin is 80% coarse"),
                             (9, "Chromatin is 90% coarse"),
                             (10, "Largest cells appear more than twice their size")]
    NormalNucleoliChoices = [(1, "Nucleoli are completely normal (small/1 per cell/barely visible)"),
                             (2, "20% of nucleoli observe some abnormality"),
                             (3, "30% of nucleoli observe some abnormality"),
                             (4, "40% of nucleoli observe some abnormality"),
                             (5, "50% of nucleoli observe some abnormality"),
                             (6, "60% of nucleoli observe some abnormality"),
                             (7, "70% of nucleoli observe some abnormality"),
                             (8, "80% of nucleoli observe some abnormality"),
                             (9, "90% of nucleoli observe some abnormality"),
                             (10, "All Nucleoli have varying degrees of abnormality")]
    MitosesChoices = [(1, "Mitotic activity is completely normal"),
                             (2, "20% of mitotic activity appears abnormal"),
                             (3, "30% of mitotic activity appears abnormal"),
                             (4, "40% of mitotic activity appears abnormal"),
                             (5, "50% of mitotic activity appears abnormal"),
                             (6, "60% of mitotic activity appears abnormal"),
                             (7, "70% of mitotic activity appears abnormal"),
                             (8, "80% of mitotic activity appears abnormal"),
                             (9, "90% of mitotic activity appears abnormal"),
                             (10, "All mitotic activity is abnormal")]
    
    ClumpThickness = forms.ChoiceField(choices=ClumpThicknessChoices, widget=forms.Select(attrs={'size': '1'}))
    UniformityofCellSize = forms.ChoiceField(choices=UniformityofCellSizeChoices)
    UniformityofCellShape = forms.ChoiceField(choices=UniformityofCellShapeChoices)
    MarginalAdhesion = forms.ChoiceField(choices=MarginalAdhesionChoices)
    SingleEpithelialCellSize = forms.ChoiceField(choices=SingleEpithelialCellSizeChoices)
    BareNuclei = forms.ChoiceField(choices=BareNucleiChoices)
    BlandChromatin = forms.ChoiceField(choices=BlandChromatinChoices)
    NormalNucleoli = forms.ChoiceField(choices=NormalNucleoliChoices)
    Mitoses = forms.ChoiceField(choices=MitosesChoices)
    