# SUMMARY:      prepare_polygon.py
# USAGE:
# ORG:          Pacific Northwest National Laboratory
# AUTHOR:       Xuehang Song
# E-MAIL:       xuehang.song@pnnl.gov
# ORIG-DATE:    06/30/2021
# DESCRIPTION:
# DESCRIPTION-END

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import joblib
from shapely import geometry
from descartes import PolygonPatch
import geopandas as gpd
#from matplotlib import colors


def read_asc(asc_file):
    """
    parse a asc file
    """

    # read file header
    with open(asc_file) as f:
        nx = int(f.readline().split("\n")[0].split(" ")[-1])
        ny = int(f.readline().split("\n")[0].split(" ")[-1])
        ox = float(f.readline().split("\n")[0].split(" ")[-1])
        oy = float(f.readline().split("\n")[0].split(" ")[-1])
        dx = dy = float(f.readline().split("\n")[0].split(" ")[-1])
        nan_value = float(f.readline().split("\n")[0].split(" ")[-1])
    # load data
    data = np.transpose(np.genfromtxt(asc_file, skip_header=6))
    data = data[:, ::-1]
    data[data == nan_value] = np.nan

    # read x,y
    x = np.arange(ox+0.5*dx, ox+dx*nx, dx)
    y = np.arange(oy+0.5*dy, oy+dy*ny, dy)

    return(x, y, data)


def plot_plume(iplume, colors, color_level):
    imgfile = img_dir+iplume+"_polys.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(site_easting, site_northing, color="white", zorder = 1) # was 1 for z order
    scatter = ax.scatter(locations[:,0], locations[:,1], c = labels_K, cmap = 'rainbow', zorder = 3000)  # Plots clusters
    ax.set_title("Well Locations labeled with clusters using CTET Aqueous Mass data")
    for i,txt in enumerate(names):
        ax.annotate(names[i].replace('-Concentration', ''), xy = (X[i], Y[i]), weight = 'bold').set_zorder(3000)
    legend1 = ax.legend(*scatter.legend_elements(),
                         loc="upper right", title="Clusters")
    ax.add_artist(legend1).set_zorder(3000)    
    ax.legend()   
    for isite in [x for x in site_polys.keys() if "200-West" in x]:            # This plots the 200 west rectangle.
        site_patch = PolygonPatch(site_polys[isite], fc='red',
                                  ec='black', fill=False,
                                  label="Management site",
                                  zorder=10000000)
        ax.add_patch(site_patch)
        ax.text(list(site_polys[isite].centroid.coords)[0][0]-2000,
                list(site_polys[isite].centroid.coords)[0][1]-1500,
                isite.split(" Area")[0],
                zorder=100000)
    for poly_index, ipoly in enumerate(plume_poly[iplume]):                    # This plots the actual plume map.
        #        print(poly_index)
        poly_patch = PolygonPatch(
            ipoly,
            fc=colors[np.where(
                color_level == plume_value[iplume][poly_index])[0][0]],
            ec=colors[np.where(
                color_level == plume_value[iplume][poly_index])[0][0]],
            fill=True,
            linewidth=0,
            label="CCI4 concentration = " +
            str(plume_value[iplume][poly_index])+"ug/L",
            zorder=1000)
        ax.add_patch(poly_patch)
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = [(h, l) for i, (h, l) in enumerate(
        zip(handles, labels)) if l not in labels[:i]]
    unique_legend = [unique_legend[0]]+[unique_legend[1:][s] for s in np.argsort(
        [float(x[1].split("=")[-1].split("ug")[0]) for x in unique_legend[1:]])]
    ax.legend(*zip(*unique_legend), loc="upper left").set_zorder(3000)
    ax.set_xlim(566000, 569000)          # ax.set_xlim(564000, 572000)  zoomed out view
    ax.set_ylim(135000, 138000)          # ax.set_ylim(133000, 142000)  
    ax.set_aspect(1)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    fig.set_size_inches(7, 7)
    fig.subplots_adjust(left=0.08,
                        right=0.98,
                        bottom=0.07,
                        top=0.96,
                        wspace=0.25,
                        hspace=0.2)
    #fig.savefig(imgfile, dpi=300, transparent=False)
    #plt.close(fig)


def plot_asc(iplume):
    imgfile = img_dir+iplume+"_asc.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(site_easting, site_northing, color="white", zorder=1)
    for isite in [x for x in site_polys.keys() if "200-West" in x]:
        site_patch = PolygonPatch(site_polys[isite], fc='red',
                                  ec='black', fill=False,
                                  label="Management site",
                                  zorder=10000000)
        ax.add_patch(site_patch)
        ax.text(list(site_polys[isite].centroid.coords)[0][0]-2000,
                list(site_polys[isite].centroid.coords)[0][1]-1500,
                isite.split(" Area")[0],
                zorder=100000)
    cf = ax.contourf(asc_data[iplume]["x"],
                     asc_data[iplume]["y"],
                     np.transpose(asc_data[iplume]["data"]),
                     #                    levels=levels,
                     zorder=10,
                     #                     norm=norm,
                     cmap=plt.cm.jet)
    ax.set_xlim(564000, 572000)
    ax.set_ylim(133000, 142000)
    ax.set_aspect(1)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    fig.set_size_inches(7, 7)
    fig.subplots_adjust(left=0.08,
                        right=0.98,
                        bottom=0.07,
                        top=0.96,
                        wspace=0.25,
                        hspace=0.2)
    fig.savefig(imgfile, dpi=300, transparent=False)
    plt.close(fig)


def plot_grid(iplume):
    imgfile = img_dir+iplume+"_check.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(site_easting, site_northing, color="white", zorder=1)
    for isite in [x for x in site_polys.keys() if "200-West" in x]:
        site_patch = PolygonPatch(site_polys[isite], fc='red',
                                  ec='black', fill=False,
                                  label="Management site",
                                  zorder=10000000)
        ax.add_patch(site_patch)
        ax.text(list(site_polys[isite].centroid.coords)[0][0]-2000,
                list(site_polys[isite].centroid.coords)[0][1]-1500,
                isite.split(" Area")[0],
                zorder=100000)
    data = plume_point[iplume].reshape(
        (len(plume_point["x"]),
         len(plume_point["y"])), order="F").T
#    data[data == 0] = np.nan

    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(color_level, len(color_level))
    cf = ax.contourf(plume_point["x"],
                     plume_point["y"],
                     data,
                     levels=color_level,
                     zorder=10,
                     norm=norm,
                     cmap=cmap)
    # ax2 = fig.add_axes([0.87, 0.12, 0.015, 0.8])
    # cb2 = plt.colorbar(cf, cax=ax2)
    # cb2.set_ticks(color_level)
    # cb2.set_ticklabels([str(int(x)) for x in color_level])
    # cb2.ax.set_ylabel("CCl4 (ug/L)",
    #                   rotation=270, labelpad=12)

    ax.set_xlim(564000, 572000)
    ax.set_ylim(133000, 142000)
    ax.set_aspect(1)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    fig.set_size_inches(7, 7)
    fig.subplots_adjust(left=0.08,
                        right=0.98,
                        bottom=0.07,
                        top=0.96,
                        wspace=0.25,
                        hspace=0.2)
    fig.savefig(imgfile, dpi=300, transparent=False)
    plt.close(fig)


def plot_plumes():
    color_level = np.unique([x for y in [plume_value[s]
                                         for s in plume_names] for x in y])
    print(color_level)
    colors = plt.cm.jet(np.linspace(0, 1, len(color_level)))
    print(colors)
    # plot plume
    for iplume in plume_names:
        plot_plume(iplume,colors,color_level)


def extract_to_txt():
    # data
    x = np.arange(565000, 572001, 10)
    y = np.arange(133000, 139001, 10)
    plume_point = dict()
    plume_point["x"] = x
    plume_point["y"] = y
    for iplume in plume_names:
        plume_point[iplume] = []
    for iy in y:
        for ix in x:
            point = geometry.Point(ix, iy)
            for iplume in plume_names:
                point_value = 0.
                for ivalue, ipoly in zip(
                        plume_value[iplume],
                        plume_poly[iplume]):
                    if ipoly.contains(point):
                        point_value = max(point_value, ivalue)
                plume_point[iplume].append(point_value)
    for iplume in plume_names:
        plume_point[iplume] = np.array(plume_point[iplume])

    # write data to file
    x_vector = x.tolist()*len(y)
    y_vector = [i for s in [[iy]*len(x) for iy in y]
                for i in s]
    output_data = [x_vector]+[y_vector] + [plume_point[x] for x in plume_names]
    output_data = np.array(output_data).T
    header = "Easting, Northing, "+", ".join(plume_names)
    np.savetxt(img_dir+"CCl4.txt", output_data, fmt="%.2f", header=header)


# load adta
data_dir = "/Users/shan594/OneDrive - PNNL/Desktop/InternshipCode-2022/InternshipCode-main/rohan_plume/Data/"
site_file = data_dir+"/Mgmt_Area_Polys.csv"
img_dir = "/Users/shan594/OneDrive - PNNL/Desktop/InternshipCode-2022/InternshipCode-main/rohan_plume/2020/"    #people/song884/all_figures/pt_testing/2020/

plume_files = [data_dir+"plumes/1/"+x for x in
               ["CTET2020_MaxBelowRLM.shp",
                "CTET2020_MaxAboveRLM.shp",
                "CTET2020_MaxFootPrint.shp"]]

plume_names = ["Ringold A", "Ringold E", "Max Footprint"]

# These joblib files are needed to plot the clusters.
locations = joblib.load('/Users/shan594/OneDrive - PNNL/Desktop/InternshipCode-2022/InternshipCode-main/Clustering_Locations.joblib')    
labels_K = joblib.load('/Users/shan594/OneDrive - PNNL/Desktop/InternshipCode-2022/InternshipCode-main/Cluster_Labels.joblib')
names = joblib.load('/Users/shan594/OneDrive - PNNL/Desktop/InternshipCode-2022/InternshipCode-main/well_names.joblib')
data = joblib.load('/Users/shan594/OneDrive - PNNL/Desktop/InternshipCode-2022/InternshipCode-main/Rohan_data/Week1/well_info.joblib')

X = data.get("x")      # Getting X and Y locations of all the wells
Y = data.get("y")    
well_name = data.get("well")

# Getting X and y locations of wells in our selected wells dataset. 
X = [];
Y = [];
for i in range(0,len(names)):
    for j in range(0,len(well_name)):
        if names[i].replace('-Concentration', '') == well_name[j]:
            X.append(data.get('x')[j])
            Y.append(data.get('y')[j])

# load site data
site_data = np.genfromtxt(site_file, delimiter=",", skip_header=1, dtype="str")
site_easting = site_data[:, 2].astype("float")
site_northing = site_data[:, 3].astype("float")
site_names = site_data[:, -1]

# create site polygons
site_points = dict()
for isite in np.unique(site_names):
    #    print(isite)
    site_points[isite] = [geometry.Point(ix, iy) for ix, iy in zip(
        site_easting[site_names == isite], site_northing[site_names == isite])]
site_polys = dict()
for isite in site_points.keys():
    site_polys[isite] = geometry.Polygon(site_points[isite])

# from shape file
shape_data = dict()
plume_poly = dict()
plume_value = dict()


for iplume, ifile in zip(plume_names, plume_files):
    shape_data[iplume] = gpd.read_file(ifile)
    plume_poly[iplume] = [geometry.Polygon(geometry.mapping(x)['coordinates'])
                          for x in shape_data[iplume].geometry]
    plume_value[iplume] = [x[0] for x in shape_data[iplume].values]

plume_data = dict()
plume_data["poly"] = plume_poly
plume_data["value"] = plume_value
#joblib.dump(plume_data, data_dir+"2020_CCl4_plume.joblib")

plot_plumes()