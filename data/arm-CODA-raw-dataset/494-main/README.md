# Arm-CODA: A dataset of upper-limb human movement during routine examination

Authors:
- [Sylvain W. Combettes](https://github.com/sylvaincom)
- [Paul Boniol](https://github.com/boniolp)
- Antoine Mazarguil
- Danping Wang
- Diego Vaquero-Ramos
- Marion Chauveau
- Laurent Oudre
- Nicolas Vayatis
- Pierre-Paul Vidal 
- Alexandra Roren
- Marie-Martine Lefevre-Colau


## How to use this repository

1. Create the clean data set (time series and metadata) by running the `create_data_and_metadata_files.ipynb` notebook.
2. Inspect the clean data set and reproduce key figures and plots of the IPOL paper by running the `reproduce_paper.ipynb` notebook.
3. Run the IPOL demo locally, for example by running in the command line:
```
$ python3 main.py --subject 4 --movement "AEBPSA" --sensor 0
```
`demo_draft.ipynb` helps understand the code from `main.py` but it is not useful: it can be discarded.

## Access the demo

The demonstration and the intercative tool can be found here:

https://ipolcore.ipol.im/demo/clientApp/demo.html?id=494

The full dataset can be found here:

https://kiwi.cmla.ens-cachan.fr/index.php/s/MrxEx99NPtGrjYx (SHASUM 256: be2696ee0908c62083af8e188a15f491a33b390724f4a37f02c20e0e7807ae8f)
The SHASUM is generated using the following command on the arm-CODA-dataset folder: tar -cf - arm-CODA-dataset | shasum -a 256
