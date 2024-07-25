# An Ordinal Latent Variable Model of Conflict Intensity


Niklas Stoehr, Lucas Torroba Hennigen, Josef Valvoda, Robert West, Ryan Cotterell, Aaron Schein<br><br>
paper pre-print at https://arxiv.org/pdf/2210.03971.pdf

## Folder structure and installation

In order to run the code locally, you have to install the user-defined modules ``g0configs``,``g1data``,``g2model`` and ``g3analysis``. From the root, install the modules by executing

```
niklasstoehr@MacBook-Pro-2 goldstein %
pip install -e g0configs
pip install -e g1data
pip install -e g2model
```

``g0configs`` features configuration methods and other helper functions <br>
``g1data`` features different data loading functionality <br>
``g2model`` features model functionality <br>
``g3analysis`` features code for visualizing, predicting and comparing against Google Trends <br>

In addition, we recommend installing the requirements listed in the requirements.txt file:
```
pip install -r requirements.txt
```


## Data

We use the publicly available Nonviolent and Violent Campaigns and Outcomes (NAVCO) data collection.
Specifically, we use the latest release [NAVCO 3.0](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/INNYEO) from November 2019 which comprises 112089 events around the world between December 1990 and December 2012. 
An exemplary event description is "On 19 May 2012, military (S) injured (P) two (Q) civilians (O) in Afghanistan."
Download the data and place it at ``data/navco/navco3-0full.xlsx``. The function ``load_navco_raw()`` in the script ``g1data/dataloading`` can then load the data for you.