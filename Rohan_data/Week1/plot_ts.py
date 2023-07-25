from scipy.stats import norm
from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt
import joblib
import re
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def compute_upper_confidence_limit(
        snapshot, cutoff_value, ucl_percentile):
    """
    Get the percentile upper confidence limit from the simulation results
    The maxium concention value in each column was use for conservative consideration
    """

    z_max_concention = np.max(
        snapshot["varis"]['aqueous plume concentration']["value"], axis=2)
    conc_samples = z_max_concention[z_max_concention >= cutoff_value]
    ucl = np.mean(conc_samples)+np.percentile(
        conc_samples,
        ucl_percentile)*np.std(conc_samples)/len(conc_samples)**0.5
    return(ucl)


def retrieve_node_value(plot, iline, nx, ny, nz):
    """
    retrieve node value
    """
    node_value = []
    while len(node_value) < nx*ny*nz:
        line_data = [float(x) for x in plot[iline].split(" ") if x]
        node_value += line_data
        iline += 1
    return node_value, iline


def length_conversion(x):
    """
    convert length units
    """
    return {
        'a': 1e-10,
        'ang': 1e-10,
        'angstrom': 1e-10,
        'ao': 1e-10,
        'cm': 0.01,
        'ffl': 109.728,
        'ft': 0.3048,
        'furlong': 201.168,
        'm': 1,
        'mi': 1609.344,
        'mile': 1609.344,
        "mm": 0.001,
        'rod': 5.0292,
        'yd': 0.9144
    }.get(x, 1)


def extract_snapshot_data(plot_file):
    """
    extract data from STOMP snapshot file
    """

    plot_dict = dict()
    with open(plot_file, "r") as f:
        plot = f.readlines()

    # remove comments and blank lines in input deck
    plot = [re.split('[#!\n]', x)[0] for x in plot]
    plot = [x.lower() for x in plot if x]

    # remove header
    n_header = [i for i, j in enumerate(plot[0:50]) if j[0:6] == "number"][0]
    plot = plot[n_header:]
    itime = int(plot[0].split(" ")[-1])
    t = float(plot[1].split(" ")[3].split(",")[0])
    plot_dict["file"] = plot_file
    plot_dict["itime"] = itime
    plot_dict["t"] = t

    # get nx,ny,nz,ox,oy,oz
    nx = int(plot[2].split(" ")[-1])
    ny = int(plot[3].split(" ")[-1])
    nz = int(plot[4].split(" ")[-1])

    # get x,y,z
    x_unit = plot[8].split(" ")[-1]
    xf, iline = retrieve_node_value(plot, 9, nx+1, ny+1, nz+1)
    xf = np.array(xf)*length_conversion(x_unit)
    y_unit = plot[iline].split(" ")[-1]
    yf, iline = retrieve_node_value(plot, iline+1, nx+1, ny+1, nz+1)
    yf = np.array(yf)*length_conversion(y_unit)
    z_unit = plot[iline].split(" ")[-1]
    zf, iline = retrieve_node_value(plot, iline+1, nx+1, ny+1, nz+1)
    zf = np.array(zf)*length_conversion(z_unit)
    plot_dict["xf"] = xf
    plot_dict["yf"] = yf
    plot_dict["zf"] = zf

    plot_dict["nx"] = nx
    plot_dict["ny"] = ny
    plot_dict["nz"] = nz

    # read variable data from plot file
    plot_dict["varis"] = dict()
    while len(plot[iline:]) > 0:
        ivari = plot[iline].split(",")[0]
        plot_dict["varis"][ivari] = dict()
        if len(plot[iline].split(",")) > 1:
            plot_dict["varis"][ivari]["unit"] = plot[iline].split(",")[-1]
        else:
            plot_dict["varis"][ivari]["unit"] = ""
        value, iline = retrieve_node_value(
            plot, iline+1, nx, ny, nz)
        plot_dict["varis"][ivari]["value"] = (
            np.array(value).reshape((nx, ny, nz), order="F"))

    return(plot_dict)


def plot_barplot(selected_wells, well_ts):
    """
    barplot of each well
    """

    imgfile = img_dir+"well_barplot.png"

    # prepare yearly data
    nyear = 8
    nwell = len(selected_wells)
    colors = cm.jet(np.linspace(0, 1, nyear))

    mass_array, aqueous_array, conc_array = clean_up_data(
        selected_wells, well_ts, nyear)

    # generate barplot
    fig = plt.figure()
    ax = plt.subplot(111)
    w = 0.3
    nwell = len(selected_wells)
    for i in range(nyear):
        ax.bar(np.arange(nwell)*w*(nyear+2)+w*i,
               mass_array[:, i],
               color=colors[i],
               label="Year "+str(i+1),
               width=0.3,
               align="center")
    ax.set_xticks(np.arange(nwell)*w*(nyear+2)+w*(nyear-1)*0.5)
    ax.set_xticklabels(selected_wells, rotation=90)
    ax.set_ylabel("CTET mass (kg)")
    plt.legend(ncol=9)
    plt.subplots_adjust(top=0.97, bottom=0.3, left=0.07,
                        right=0.99, wspace=0.55, hspace=0.4)
    fig.set_size_inches(12, 4)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    # generate aqueous barplot
    imgfile = img_dir+"aqueous_barplot.png"
    # generate barplot
    fig = plt.figure()
    ax = plt.subplot(111)
    w = 0.3
    for i in range(nyear):
        ax.bar(np.arange(nwell)*w*(nyear+2)+w*i,
               aqueous_array[:, i],
               color=colors[i],
               label="Year "+str(i+1),
               width=0.3,
               align="center")
    ax.set_xticks(np.arange(nwell)*w*(nyear+2)+w*(nyear-1)*0.5)
    ax.set_xticklabels(selected_wells, rotation=90)
    ax.set_ylabel("Mean aqueous mass (GPM)")
    ax.set_ylim(0, 160)
    plt.legend(ncol=9)
    plt.subplots_adjust(top=0.97, bottom=0.3, left=0.07,
                        right=0.99, wspace=0.55, hspace=0.4)
    fig.set_size_inches(12, 4)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    # generate aqueous barplot
    imgfile = img_dir+"conc_barplot.png"
    # generate barplot
    fig = plt.figure()
    ax = plt.subplot(111)
    w = 0.3
    nwell = len(selected_wells)
    for i in range(nyear):
        ax.bar(np.arange(nwell)*w*(nyear+2)+w*i,
               conc_array[:, i],
               color=colors[i],
               label="Year "+str(i+1),
               width=0.3,
               align="center")
    ax.set_xticks(np.arange(nwell)*w*(nyear+2)+w*(nyear-1)*0.5)
    ax.set_xticklabels(selected_wells, rotation=90)
    ax.set_ylabel("Mean CTET concentration (ug/L)")
    ax.set_ylim(0, 2000)
    plt.legend(ncol=9)
    plt.subplots_adjust(top=0.97, bottom=0.3, left=0.07,
                        right=0.99, wspace=0.55, hspace=0.4)
    fig.set_size_inches(12, 4)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_histgram(selected_wells):
    """
    barplot of each well
    """

    imgfile = img_dir+"well_histgram.png"

    # prepare yearly data
    nyear = 8
    mass_array = np.zeros((len(selected_wells), nyear))
    colors = cm.jet(np.linspace(0, 1, nyear))
    for i, iwell in enumerate(selected_wells):
        mass_data = np.array(well_ts[iwell]['ctet_mass'])
        mass_data = mass_data[mass_data > 0]
        for iyear in range(int(len(mass_data)/12)):
            mass_array[i, iyear] = np.sum(mass_data[np.arange(12)+iyear*12])
    selected_wells = np.array(selected_wells)[
        np.argsort(mass_array[:, 0])[::-1]]
    mass_array = mass_array[np.argsort(mass_array[:, 0])[::-1], :]

    # generate histgram
    ncol = 4
    nrow = int(np.ceil(nyear/ncol))
    fig, axs = plt.subplots(nrow, ncol)

    for i, ax in enumerate(fig.axes[nyear:]):
        ax.set_axis_off()
    for i, ax in enumerate(fig.axes[0: nyear]):
        ax.hist(mass_array[:, i][mass_array[:, i] > 0],
                bins=np.arange(0, 300, 15),
                density=True)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 0.02)
        ax.set_title(iwell, fontsize=15)
        # ax.set_ylim(-0.1*max_mass, max_mass*1.1)
        # ax.set_xlim(-0.5, 8.5)
        # ax.set_ylabel("Cumulative CTET mass (kg)", fontsize=10, color="red")
        # ax.set_xlabel("Time (yr)", fontsize=10)
        # ax.spines['left'].set_color('red')
        # ax.tick_params(axis='y', colors='red')
        # ax.tick_params(axis="x", which="major",
        #                labelsize=10, labelrotation=0)
    plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04,
                        right=0.97, wspace=0.55, hspace=0.45)
    fig.set_size_inches(16, 8)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_two_year(selected_wells, well_ts):
    """
    """

    # prepare yearly data
    nyear = 8
    nwell = len(selected_wells)
    colors = cm.tab20(np.linspace(0, 1, nwell))

    mass_array, aqueous_array, conc_array = clean_up_data(
        selected_wells, well_ts, nyear)

    imgfile = img_dir+"mass_two_year.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, iwell in enumerate(selected_wells):
        mass_data = mass_array[i, :][mass_array[i, :] > 0]
        x = mass_data[:-1]
        y = mass_data[1:]
        ax.scatter(x, y,
                   facecolors="None",
                   edgecolors=colors[i],
                   lw=3)
    ax.plot([0, 350], [0, 350], color="black")
    ax.set_ylim(0, 350)
    ax.set_xlim(0, 350)
    ax.set_ylabel("CTET mass recovery in Year N+1 (Kg)")
    ax.set_xlabel("CTET mass recovery in Year N (Kg)")
    ax.set_xlim(0, 350)
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.12,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(6, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    imgfile = img_dir+"aqueous_two_year.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, iwell in enumerate(selected_wells):
        aqueous_data = aqueous_array[i, :][mass_array[i, :] > 0]
        x = aqueous_data[:-1]
        y = aqueous_data[1:]
        ax.scatter(x, y,
                   facecolors="None",
                   edgecolors=colors[i],
                   lw=3)
    ax.plot([50, 150], [50, 150], color="black")
    ax.set_ylim(50, 150)
    ax.set_xlim(50, 150)
    ax.set_ylabel("Averaged pumping rate in Year N+1 (GPM)")
    ax.set_xlabel("Averaged pumping rate in Year N (Kg)")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.12,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(6, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    imgfile = img_dir+"conc_two_year.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    all_x = []
    all_y = []
    for i, iwell in enumerate(selected_wells):
        conc_data = conc_array[i, :][mass_array[i, :] > 0]
        x = conc_data[:-1]
        y = conc_data[1:]
        all_x.append(x)
        all_y.append(y)
        ax.scatter(x, y,
                   facecolors="None",
                   edgecolors=colors[i],
                   lw=3)
    all_x = np.array([x for y in all_x for x in y])
    all_y = np.array([x for y in all_y for x in y])
    linear_fit = np.polyfit(all_x, all_y, 1, full=True)
    m, b = linear_fit[0]
    residuals = linear_fit[1][0]
    r_square = (1-residuals/np.sum((all_y-np.mean(all_y))**2))
    ax.text(100, 1600,
            ("y="+"{:5.3f}".format(m)+"x+" +
             "{:5.3f}".format(b) +
             "; $R^2$=" +
             "{:5.3f}".format(r_square)),
            fontsize=15, color='red')
    ax.plot(np.linspace(0, 2000, 100), m *
            np.linspace(0, 2000, 100) + b, color="red")
    ax.plot([0, 2000], [0, 2000], color="black")
    ax.set_ylim(0, 2000)
    ax.set_xlim(0, 2000)
    ax.set_ylabel("Averaged CTET concentration in Year N+1 (ug/L)")
    ax.set_xlabel("Averaged CTET concentration in Year N (ug/L)")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.12,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(6, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    imgfile = img_dir+"conc_two_year_diff2.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, iwell in enumerate(selected_wells):
        conc_data = conc_array[i, :][conc_array[i, :] > 0]
        x = conc_data[:-1]
        y = conc_data[1:]
        ax.scatter(x, y-(m*x+b),
                   facecolors="None",
                   edgecolors=colors[i],
                   lw=3)
    ax.set_xlabel("CTET mean concentration [ug/L]")
    ax.plot([0, 2000], [0, 0], color="black")
    ax.plot([1000, 1000], [-1, 1], color="black")
    ax.set_ylabel("$S_{N+1}$-(m*$S_{N}$+b) [ug/L]")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.12,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(6, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    imgfile = img_dir+"conc_two_year_hist2.png"
    # generate histgram
    fig = plt.figure()
    ax = plt.subplot(111)
    data = all_y-(m*all_x+b)
    mean, std = norm.fit(data)
    ax.hist(data,
            bins=np.linspace(-500, 500, 20), fc="None", ec="black", density=True)
    ax.set_xlabel("$S_{N+1}$-(m*$S_{N}$+b) [ug/L]")
    ax.set_ylabel("pdf")
    ax.set_xlim(-500, 500)
    x = np.linspace(-500, 500, 100)
    y = norm.pdf(x, mean, std)
    ax.plot(x, y, color="red")
    ax.text(-400, 0.004,
            "Mean = "+"{:5.3f}".format(mean),
            fontsize=15, color='red')
    ax.text(-400, 0.0037,
            "STD = "+"{:5.3f}".format(std),
            fontsize=15, color='red')

    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.12,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(6, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    imgfile = img_dir+"conc_two_year4.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    all_x = []
    all_y = []
    for i, iwell in enumerate(selected_wells):
        conc_data = conc_array[i, :][mass_array[i, :] > 0]
        x = conc_data[:-1]
        y = conc_data[1:]
        all_x.append(x)
        all_y.append(y)
        ax.scatter(x, y,
                   facecolors="None",
                   edgecolors=colors[i],
                   lw=3)
    all_x = np.array([x for y in all_x for x in y])
    all_y = np.array([x for y in all_y for x in y])

    # fit #1
    all_x_1 = all_x[all_x <= 1000]
    all_y_1 = all_y[all_x <= 1000]
    linear_fit_1 = np.polyfit(
        all_x_1, all_y_1, 1, full=True)
    m_1, b_1 = linear_fit_1[0]
    residuals_1 = linear_fit_1[1][0]
    r_square_1 = (1-residuals_1/np.sum((all_y_1-np.mean(all_y_1))**2))

    # fit #2
    all_x_2 = all_x[all_x > 1000]
    all_y_2 = all_y[all_x > 1000]
    linear_fit_2 = np.polyfit(
        all_x_2, all_y_2, 1, full=True)
    m_2, b_2 = linear_fit_2[0]
    residuals_2 = linear_fit_2[1][0]
    r_square_2 = (1-residuals_2/np.sum((all_y_2-np.mean(all_y_2))**2))

    ax.text(100, 1600,
            ("y="+"{:5.3f}".format(m_1)+"x+" +
             "{:5.3f}".format(b_1) +
             "; $R^2$=" +
             "{:5.3f}".format(r_square_1)),
            fontsize=15, color='red')
    ax.plot(np.linspace(0, 1000, 100), m_1 *
            np.linspace(0, 1000, 100) + b_1,
            lw=2,
            color="red")

    ax.text(100, 1400,
            ("y="+"{:5.3f}".format(m_2)+"x+" +
             "{:5.3f}".format(b_2) +
             "; $R^2$=" +
             "{:5.3f}".format(r_square_2)),
            fontsize=15, color='green')
    ax.plot(np.linspace(1000, 2000, 100), m_2 *
            np.linspace(1000, 2000, 100) + b_2,
            color="green",
            lw=2)

    ax.plot([0, 2000], [0, 2000], color="black")
    ax.set_ylim(0, 2000)
    ax.set_xlim(0, 2000)
    ax.set_ylabel("Averaged CTET concentration in Year N+1 (ug/L)")
    ax.set_xlabel("Averaged CTET concentration in Year N (ug/L)")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.12,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(6, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_two_month(selected_wells):
    """
    """
    nwell = len(selected_wells)
    colors = cm.jet(np.linspace(0, 1, nwell))

    imgfile = img_dir+"well_two_month.png"

    # generate histgram
    fig = plt.figure()
    ax = plt.subplot(111)

    for i, iwell in enumerate(selected_wells):
        mass_data = well_ts[iwell]['ctet_mass'][well_ts[iwell]
                                                ['ctet_mass'] > 0]
        x = np.array(mass_data[:-1])
        y = np.array(mass_data[1:])
        print((y-x)/x)
        ax.scatter(x, (y-x)/x,
                   facecolors="None",
                   edgecolors=colors[i],
                   lw=3)
    # ax.plot([-5, 50], [-5, 50], color="black")
#    ax.set_ylim(-5, 50)
    ax.set_ylim(-30, 30)
    # ax.set_xlim(-5, 50)
    ax.set_ylabel("CTET Mass recovery in Year N+1 (Kg)")
    ax.set_xlabel("CTET Mass recovery in Year N (Kg)")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.1,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(8, 8)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_two_year_hist(selected_wells):
    """
    """

    # prepare yearly data
    nyear = 8
    mass_array = np.zeros((len(selected_wells), nyear))
    colors = cm.jet(np.linspace(0, 1, nyear))
    for i, iwell in enumerate(selected_wells):
        mass_data = np.array(well_ts[iwell]['ctet_mass'])
#        mass_data = mass_data[mass_data > 0]
#        mass_data = mass_data[np.where(np.abs(mass_data) > 0)[0][0]:]
        mass_data = mass_data[np.abs(
            np.array(well_ts[iwell]['aqueous'])) > 50]

        for iyear in range(int(len(mass_data)/12)):
            mass_array[i, iyear] = np.sum(mass_data[np.arange(12)+iyear*12])

    selected_wells = np.array(selected_wells)[
        np.argsort(mass_array[:, 0])[::-1]]
    mass_array = mass_array[np.argsort(mass_array[:, 0])[::-1], :]

    # get all data
    year_1 = []
    year_2 = []
    for i, iwell in enumerate(selected_wells):
        mass_data = mass_array[i, :][mass_array[i, :] > 0]
        year_1.append(mass_data[:-1])
        year_2.append(mass_data[1:])
    year_1 = np.array([m for n in year_1 for m in n])
    year_2 = np.array([m for n in year_2 for m in n])

    imgfile = img_dir+"well_two_year_hist1.png"
    # generate histgram
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.hist(year_2-year_1, bins=25, fc="None", ec="black", density=True)
#    ax.set_xlim(-150, 150)
    ax.set_xlabel("$S_{N}$-$S_{N-1}$ [Kg]")
    ax.set_ylabel("pdf")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.1,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(8, 8)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    imgfile = img_dir+"well_two_year_hist2.png"
    # generate histgram
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.hist((year_2-year_1)/year_1, bins=25,
            fc="None", ec="black", density=True)
#    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("($S_{N}$-$S_{N-1}$)/($S_{N}$+$S_{N-1}$) [-]")
    ax.set_ylabel("pdf")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.1,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(8, 8)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    imgfile = img_dir+"well_two_year_hist3.png"
    # axs
    fig = plt.figure()
    ax = plt.subplot(111)
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    ax2 = axs[1]

    # generate histgram
    data = ((year_2-year_1)/year_1)[year_1 <= 150]
    print(np.mean(data))
    ax.hist(data, bins=10,
            fc="None", ec="black", density=True)
#    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("($S_{N}$-$S_{N-1}$)/($S_{N}$+$S_{N-1}$) [-]")
    ax.set_ylabel("pdf")

    # generate histgram
    data = ((year_2-year_1)/year_1)[year_1 > 150]
    print(np.mean(data))
    ax2.hist(data, bins=10,
             fc="None", ec="black", density=True)
#    ax2.set_xlim(-1.5, 1.5)
    ax2.set_xlabel("($S_{N}$-$S_{N-1}$)/($S_{N}$+$S_{N-1}$) [-]")
    ax2.set_ylabel("pdf")
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15,
                        right=0.97, wspace=0.3, hspace=0.45)
    fig.set_size_inches(9, 4)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_diff(selected_wells, well_ts):
    # prepare yearly data
    nyear = 8
    nwell = len(selected_wells)
    colors = cm.tab20(np.linspace(0, 1, nwell))

    mass_array, aqueous_array, conc_array = clean_up_data(
        selected_wells, well_ts, nyear)

    imgfile = img_dir+"conc_two_year_diff0.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, iwell in enumerate(selected_wells):
        conc_data = conc_array[i, :][conc_array[i, :] > 0]
        x = conc_data[:-1]
        y = conc_data[1:]
        ax.scatter(x, y-x,
                   facecolors="None",
                   edgecolors=colors[i],
                   lw=3)
    ax.set_xlabel("CTET mean concentration [ug/L]")
    ax.set_xlim(0, 2000)
    ax.set_ylim(-800, 800)
    ax.plot([0, 2000], [0, 0], color="black")
    ax.plot([1000, 1000], [-1000, 1000], color="black")
    ax.set_ylabel("$C_{N+1}$-$C_{N}$ [ug/L]")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.12,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(6, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    # collect all data
    all_x = []
    all_y = []
    for i, iwell in enumerate(selected_wells):
        conc_data = conc_array[i, :][conc_array[i, :] > 0]
        x = conc_data[:-1]
        y = conc_data[1:]
        all_x.append(x)
        all_y.append(y)
    all_x = np.array([x for y in all_x for x in y])
    all_y = np.array([x for y in all_y for x in y])

    imgfile = img_dir+"conc_two_year_hist0.png"
    # generate histgram
    fig = plt.figure()
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    ax2 = axs[1]

    data = (all_y-all_x)[all_x <= 1000]
    mean, std = norm.fit(data)
    ax.hist(data,
            bins=np.linspace(-800, 800, 40),
            fc="None", ec="black", density=True)
    x = np.linspace(-800, 800, 40)
    y = norm.pdf(x, mean, std)
    ax.plot(x, y, color="red")
    ax.set_xlabel("($C_{N+1}$-$C_{N}$) [ug/L]")
    ax.set_ylabel("pdf")
    ax.set_title("$C_{N}$<=1000 [ug/L]")

    data = (all_y-all_x)[all_x > 1000]
    mean, std = norm.fit(data)
    ax2.hist(data,
             bins=np.linspace(-800, 800, 40),
             fc="None", ec="black", density=True)
    x = np.linspace(-800, 800, 40)
    y = norm.pdf(x, mean, std)
    ax2.plot(x, y, color="red")
    ax2.set_xlabel("($C_{N+1}$-$C_{N}$) [ug/L]")
    ax2.set_ylabel("pdf")
    ax2.set_title("$C_{N}$>1000 [ug/L]")

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.06,
                        right=0.97, wspace=0.2, hspace=0.4)
    fig.set_size_inches(12, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    imgfile = img_dir+"conc_two_year_diff1.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, iwell in enumerate(selected_wells):
        conc_data = conc_array[i, :][conc_array[i, :] > 0]
        x = conc_data[:-1]
        y = conc_data[1:]
        ax.scatter(x, (y-x)/x,
                   facecolors="None",
                   edgecolors=colors[i],
                   lw=3)
    ax.set_xlabel("CTET mean concentration [ug/L]")
    ax.set_xlim(0, 2000)
    ax.set_ylim(-0.8, 0.8)
    ax.plot([0, 2000], [0, 0], color="black")
    ax.plot([1000, 1000], [-1, 1], color="black")
    ax.set_ylabel("($C_{N+1}$-$C_{N}$)/$C_{N}$ [ug/L]")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.12,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(6, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    # collect all data
    all_x = []
    all_y = []
    for i, iwell in enumerate(selected_wells):
        conc_data = conc_array[i, :][conc_array[i, :] > 0]
        x = conc_data[:-1]
        y = conc_data[1:]
        all_x.append(x)
        all_y.append(y)
    all_x = np.array([x for y in all_x for x in y])
    all_y = np.array([x for y in all_y for x in y])

    imgfile = img_dir+"conc_two_year_hist1.png"
    # generate histgram
    fig = plt.figure()
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    ax2 = axs[1]

    data = ((all_y-all_x)/all_x)[all_x <= 1000]
    mean, std = norm.fit(data)
    ax.hist(data,
            bins=np.linspace(-0.8, 0.8, 40),
            fc="None", ec="black", density=True)
    x = np.linspace(-0.8, 0.8, 100)
    y = norm.pdf(x, mean, std)
    ax.plot(x, y, color="red")
    ax.set_xlabel("($C_{N+1}$-$C_{N}$)/$C_{N}$ [-]")
    ax.set_ylabel("pdf")
    ax.set_title("$C_{N}$<=1000 [ug/L]")

    data = ((all_y-all_x)/all_x)[all_x > 1000]
    mean, std = norm.fit(data)
    ax2.hist(data,
             bins=np.linspace(-0.8, 0.8, 40),
             fc="None", ec="black", density=True)
    x = np.linspace(-0.8, 0.8, 100)
    y = norm.pdf(x, mean, std)
    ax2.plot(x, y, color="red")
    ax2.set_xlabel("($C_{N+1}$-$C_{N}$)/$C_{N}$ [-]")
    ax2.set_ylabel("pdf")
    ax2.set_title("$C_{N}$>1000 [ug/L]")

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.06,
                        right=0.97, wspace=0.2, hspace=0.4)
    fig.set_size_inches(12, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def clean_up_data(selected_wells, well_ts, nyear):
    """
    get yearly data
    """
    nyear = 8
    nmonth = 12
  # ctet_mass
    mass_array = np.zeros((len(selected_wells), nyear))
    for i, iwell in enumerate(selected_wells):
        mass_data = np.array(well_ts[iwell]['ctet_mass'])
        mass_data = mass_data[np.abs(
            np.array(well_ts[iwell]['aqueous'])) > 50]
        for iyear in range(int(len(mass_data)/nmonth)):
            mass_array[i, iyear] = np.sum(
                mass_data[np.arange(nmonth)+iyear*nmonth])
    selected_wells = np.array(selected_wells)[
        np.argsort(mass_array[:, 0])[::-1]]
    mass_array = mass_array[np.argsort(mass_array[:, 0])[::-1], :]

    # aqeous array
    aqueous_array = np.zeros((len(selected_wells), nyear))
    for i, iwell in enumerate(selected_wells):
        aqueous_data = np.abs(np.array(well_ts[iwell]['aqueous']))
        aqueous_data = aqueous_data[np.abs(
            np.array(well_ts[iwell]['aqueous'])) > 50]
        for iyear in range(int(len(aqueous_data)/nmonth)):
            aqueous_array[i, iyear] = np.mean(
                aqueous_data[np.arange(nmonth)+iyear*nmonth])

    # conc array
    conc_array = np.zeros((len(selected_wells), nyear))
    colors = cm.jet(np.linspace(0, 1, nyear))
    for i, iwell in enumerate(selected_wells):
        conc_data = np.abs(np.array(well_ts[iwell]['ctet_conc']))
        conc_data = conc_data[np.abs(
            np.array(well_ts[iwell]['aqueous'])) > 50]
        for iyear in range(int(len(conc_data)/nmonth)):
            conc_array[i, iyear] = np.mean(
                conc_data[np.arange(nmonth)+iyear*nmonth])

    return(mass_array, aqueous_array, conc_array)


def plot_relation():

    imgfile = img_dir+"conc_mass_relation.png"

    colors = cm.jet(np.linspace(0, 1, len(selected_wells)))

    fig = plt.figure()
    ax = plt.subplot(111)

    for i, iwell in enumerate(selected_wells):
        x = well_ts[iwell]["ctet_conc"][well_ts[iwell]["aqueous"] != 0]
        y = well_ts[iwell]["ctet_mass"][well_ts[iwell]["aqueous"] != 0]
        ax.scatter(x, y,
                   facecolors="None",
                   edgecolors=colors[i],
                   lw=3)
        ax.set_ylabel("CTET mass (kg)")
        ax.set_xlabel("CTET concentration (ug/L)")
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.1,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(8, 8)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_relation_all():

    imgfile = img_dir+"conc_mass_relation_all.png"
    nwell = len(selected_wells)
    ncol = 5
    nrow = int(np.ceil(nwell/ncol))
    line_handles = {}

    max_mass = 0
    max_aqueous = 0
    max_conc = 0
    for iwell in selected_wells:
        max_mass = max(max_mass, np.max(well_ts[iwell]["ctet_mass"]))
        max_aqueous = max(max_aqueous, np.max(
            np.abs(well_ts[iwell]["aqueous"])))
        max_conc = max(max_conc, np.max(
            np.abs(well_ts[iwell]["ctet_conc"])))

    fig = plt.figure()
    ax = plt.subplot(111)

    fig, axs = plt.subplots(nrow, ncol)

    for i, ax in enumerate(fig.axes[nwell:]):
        ax.set_axis_off()

    for i, ax in enumerate(fig.axes[0: nwell]):
        iwell = selected_wells[i]
        x = well_ts[iwell]["ctet_conc"][well_ts[iwell]["aqueous"] != 0]
        y = well_ts[iwell]["ctet_mass"][well_ts[iwell]["aqueous"] != 0]
        ax.scatter(x, y,
                   facecolors="None",
                   edgecolors="black",
                   lw=1)
        ax.set_title(iwell, fontsize=15)
        ax.set_xlim(-0.1*max_conc, max_conc*1.1)
        ax.set_ylim(-0.1*max_mass, max_mass*1.1)
        ax.set_ylabel("CTET mass (kg)")
        ax.set_xlabel("CTET concentration (ug/L)")

    plt.subplots_adjust(top=0.97, bottom=0.04, left=0.04,
                        right=0.97, wspace=0.3, hspace=0.3)
    fig.set_size_inches(20, 16)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_cumulative_all():

    imgfile = img_dir+"conc_mass_cumulative_all.png"
    nwell = len(selected_wells)
    ncol = 5
    nrow = int(np.ceil(nwell/ncol))
    line_handles = {}

    max_mass = 0
    max_aqueous = 0
    max_conc = 0
    t_min = np.min(well_ts[selected_wells[0]]["times"])
    t_max = np.max(well_ts[selected_wells[0]]["times"])

    for iwell in selected_wells:
        max_mass = max(max_mass, np.sum(np.abs(well_ts[iwell]["ctet_mass"])))
        max_aqueous = max(max_aqueous, np.sum(
            np.abs(well_ts[iwell]["aqueous"])))
        t_min = min(t_min, np.min(
            well_ts[iwell]["times"]))
        t_max = max(t_max, np.max(
            well_ts[iwell]["times"]))

    fig = plt.figure()
    ax = plt.subplot(111)

    fig, axs = plt.subplots(nrow, ncol)

    for i, ax in enumerate(fig.axes[nwell:]):
        ax.set_axis_off()

    for i, ax in enumerate(fig.axes[0: nwell]):

        iwell = selected_wells[i]
        x = well_ts[iwell]["times"]
        y = np.cumsum(well_ts[iwell]["ctet_mass"])
        x = x[y > 0]
        y = y[y > 0]
        ax.plot(x,
                y,
                color='red',
                zorder=100,
                label=iwell)
        ax.set_title(iwell, fontsize=15)
        ax.set_ylim(-0.1*max_mass, max_mass*1.1)
        ax.set_xlim(t_min, t_max)
        ax.set_ylabel("Cumulative CTET mass (kg)", fontsize=10, color="red")
        ax.spines['left'].set_color('red')
        ax.tick_params(axis='y', colors='red')
        ax.tick_params(axis="x", which="major",
                       labelsize=10, labelrotation=45)

        # aqueous
        ax2 = ax.twinx()
        x = well_ts[iwell]["times"]
        y = np.cumsum(np.abs(well_ts[iwell]["aqueous"]))
        x = x[y > 0]
        y = y[y > 0]
        ax2.plot(x,
                 y,
                 color="blue")
        ax2.set_ylim(-0.1*max_aqueous, max_aqueous*1.1)
        ax2.set_xlim(t_min, t_max)
        ax2.tick_params(axis='y', colors='blue')
        ax2.spines['right'].set_color('blue')
        ax2.set_ylabel("cumulative aqueous mass (GPM)",
                       fontsize=10, color="blue")

    plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04,
                        right=0.97, wspace=0.55, hspace=0.45)
    fig.set_size_inches(20, 10)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_cumulative_all_delta():

    imgfile = img_dir+"conc_mass_cumulative_all_delta.png"
    nwell = len(selected_wells)
    ncol = 5
    nrow = int(np.ceil(nwell/ncol))
    line_handles = {}

    max_mass = 0
    max_aqueous = 0
    max_conc = 0

    for iwell in selected_wells:
        max_mass = max(max_mass, np.sum(np.abs(well_ts[iwell]["ctet_mass"])))
        max_aqueous = max(max_aqueous, np.sum(
            np.abs(well_ts[iwell]["aqueous"])))

    fig = plt.figure()
    ax = plt.subplot(111)

    fig, axs = plt.subplots(nrow, ncol)

    for i, ax in enumerate(fig.axes[nwell:]):
        ax.set_axis_off()

    for i, ax in enumerate(fig.axes[0: nwell]):

        iwell = selected_wells[i]
        y = np.cumsum(well_ts[iwell]["ctet_mass"])
        y = y[y > 0]
        ax.plot(np.arange(len(y))/12,
                y,
                color='red',
                zorder=100,
                label=iwell)
        ax.set_title(iwell, fontsize=15)
        ax.set_ylim(-0.1*max_mass, max_mass*1.1)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylabel("Cumulative CTET mass (kg)", fontsize=10, color="red")
        ax.set_xlabel("Time (yr)", fontsize=10)
        ax.spines['left'].set_color('red')
        ax.tick_params(axis='y', colors='red')
        ax.tick_params(axis="x", which="major",
                       labelsize=10, labelrotation=0)

        # aqueous
        ax2 = ax.twinx()
        y = np.cumsum(np.abs(well_ts[iwell]["aqueous"]))
        y = y[y > 0]
        ax2.plot(np.arange(len(y))/12,
                 y,
                 color="blue")
        ax2.set_xlim(-0.5, 9.8)
        ax2.set_ylim(-0.1*max_aqueous, max_aqueous*1.1)
        ax2.tick_params(axis='y', colors='blue')
        ax2.spines['right'].set_color('blue')
        ax2.set_ylabel("cumulative aqueous mass (GPM)",
                       fontsize=10, color="blue")

    plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04,
                        right=0.97, wspace=0.55, hspace=0.45)
    fig.set_size_inches(20, 10)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_cumulative_delta():

    imgfile = img_dir+"conc_mass_cumulative_delta.png"
    nwell = len(selected_wells)
    ncol = 5
    nrow = int(np.ceil(nwell/ncol))
    line_handles = {}

    max_mass = 0
    max_aqueous = 0
    max_conc = 0

    for iwell in selected_wells:
        max_mass = max(max_mass, np.sum(np.abs(well_ts[iwell]["ctet_mass"])))
        max_aqueous = max(max_aqueous, np.sum(
            np.abs(well_ts[iwell]["aqueous"])))

    fig = plt.figure()
    ax = plt.subplot(111)
    fig, axs = plt.subplots(1, 2)

    ax = axs[0]
    ax2 = axs[1]

    for i in range(nwell):

        iwell = selected_wells[i]

        # aqueous
        y = np.cumsum(np.abs(well_ts[iwell]["aqueous"]))
        y = y[y > 0]
        ax.plot(np.arange(len(y))/12,
                y)

        # mass
        y = np.cumsum(well_ts[iwell]["ctet_mass"])
        y = y[y > 0]
        ax2.plot(np.arange(len(y))/12,
                 y,
                 zorder=100,
                 label=iwell)

    ax.set_title("Aqueous mass", fontsize=15)
    ax.set_ylim(-0.1*max_aqueous, max_aqueous*1.1)
    ax.set_xlim(-0.5, 8.9)
    ax.set_ylabel("cumulative aqueous mass (GPM)",
                  fontsize=10)
    ax.set_xlabel("Time (yr)", fontsize=10)

    ax2.set_title("CTET mass", fontsize=15)
    ax2.set_xlim(-0.5, 8.9)
    ax2.set_ylim(-0.1*max_mass, max_mass*1.1)
    ax2.set_xlabel("Time (yr)", fontsize=10)
    ax2.set_ylabel("Cumulative CTET mass (kg)", fontsize=10)
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15,
                        right=0.97, wspace=0.3, hspace=0.45)
    fig.set_size_inches(9, 4)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_filled_ts(selected_wells, well_ts):
    """
    plot time series
    """
    m = 0.808
    b = 69.997
    std = 128.544

    nyear = 8
    nwell = len(selected_wells)
    colors = cm.jet(np.linspace(0, 1, nyear))

    mass_array, aqueous_array, conc_array = clean_up_data(
        selected_wells, well_ts, nyear)

    ncol = 5
    nrow = int(np.ceil(nwell/ncol))
    imgfile = img_dir+"well_ts_filled.png"
    fig, axs = plt.subplots(nrow, ncol)

    for i, ax in enumerate(fig.axes[nwell:]):
        ax.set_axis_off()

    for i, ax in enumerate(fig.axes[0: nwell]):
        iwell = selected_wells[i]
        ori_data = conc_array[i, :][conc_array[i, :] > 0]
        ax.plot(np.arange(len(ori_data)),
                ori_data,
                color="orange",
                label=iwell)
        ndata = 10
        for iline in range(ndata):
            new_data = [ori_data[-1]]
            for idata in range(10):
                new_data.append(new_data[-1]*m+b -
                                np.random.normal(0, std)*0.2)
            ax.plot(np.append(len(ori_data)-1, len(ori_data)+np.arange(ndata)),
                    new_data,
                    color="grey",
                    lw=0.5,
                    label=iwell)
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 2000)
        if iwell in selected_wells:
            ax.set_title(iwell, fontsize=15, color="red", fontweight="bold")
        else:
            ax.set_title(iwell, fontsize=15)
        ax.set_xlabel("Time (yr)",
                      fontsize=10)
        ax.set_ylabel("CTET concentration (ug/L)",
                      fontsize=10)
        ax.tick_params(axis="x", which="major",
                       labelsize=10)

    plt.subplots_adjust(top=0.97, bottom=0.04, left=0.07,
                        right=0.9, wspace=0.55, hspace=0.4)
    fig.set_size_inches(30, 15)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_ts():
    """
    plot time series
    """

    wells = list(well_ts.keys())
    wells = [x for x in wells if "YE" in well_ts[x]["id"]]
    line_handles = {}

    max_mass = 0
    max_aqueous = 0
    max_conc = 0
    for iwell in wells:
        max_mass = max(max_mass, np.max(well_ts[iwell]["ctet_mass"]))
        max_aqueous = max(max_aqueous, np.max(
            np.abs(well_ts[iwell]["aqueous"])))
        max_conc = max(max_conc, np.max(
            np.abs(well_ts[iwell]["ctet_conc"])))

    nwell = len(wells)
    ncol = 5
    nrow = int(np.ceil(nwell/ncol))
    imgfile = img_dir+"well_ts.png"
    fig, axs = plt.subplots(nrow, ncol)

    for i, ax in enumerate(fig.axes[nwell:]):
        ax.set_axis_off()

    for i, ax in enumerate(fig.axes[0: nwell]):
        iwell = wells[i]
        ax.plot(
            well_ts[iwell]["times"],
            well_ts[iwell]["ctet_mass"],
            color='red',
            zorder=100,
            label=iwell)
        if iwell in selected_wells:
            ax.set_title(iwell, fontsize=15, color="red", fontweight="bold")
        else:
            ax.set_title(iwell, fontsize=15)

        ax.set_ylim(-0.1*max_mass, max_mass*1.1)
        ax.set_ylabel("CTET mass (kg)", fontsize=10, color="red")
        ax.spines['left'].set_color('red')
        ax.tick_params(axis='y', colors='red')
        ax.tick_params(axis="x", which="major",
                       labelsize=10, labelrotation=45)

        # aqueous
        ax2 = ax.twinx()
        ax2.plot(well_ts[iwell]["times"],
                 np.abs(well_ts[iwell]["aqueous"]),
                 color="blue")
        ax2.set_ylim(-0.1*max_aqueous, max_aqueous*1.1)
        ax2.tick_params(axis='y', colors='blue')
        ax2.spines['right'].set_color('blue')
        ax2.set_ylabel("Aqueous mass (GPM)",
                       fontsize=10, color="blue")

        # conc
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        ax3.plot(well_ts[iwell]["times"],
                 np.abs(well_ts[iwell]["ctet_conc"]),
                 color="orange")
        ax3.set_ylim(-0.1*max_conc, max_conc*1.1)
        ax3.tick_params(axis='y', colors='orange')
        ax3.spines['right'].set_color('orange')
        ax3.set_ylabel("CTET concentration (ug/L)",
                       fontsize=10, color="orange")
    plt.subplots_adjust(top=0.97, bottom=0.04, left=0.07,
                        right=0.9, wspace=0.55, hspace=0.4)
    fig.set_size_inches(30, 20)
    fig.savefig(imgfile, bbox_inches=0)

    plt.close(fig)

    nwell = len(selected_wells)
    ncol = 5
    nrow = int(np.ceil(nwell/ncol))
    imgfile = img_dir+"well_ts_small.png"
    fig, axs = plt.subplots(nrow, ncol)

    for i, ax in enumerate(fig.axes[nwell:]):
        ax.set_axis_off()

    for i, ax in enumerate(fig.axes[0: nwell]):
        iwell = selected_wells[i]
        ax.plot(
            well_ts[iwell]["times"],
            well_ts[iwell]["ctet_mass"],
            color='red',
            zorder=100,
            label=iwell)
        if iwell in selected_wells:
            ax.set_title(iwell, fontsize=15, color="red", fontweight="bold")
        else:
            ax.set_title(iwell, fontsize=15)

        ax.set_ylim(-0.1*max_mass, max_mass*1.1)
        ax.set_ylabel("CTET mass (kg)", fontsize=10, color="red")
        ax.spines['left'].set_color('red')
        ax.tick_params(axis='y', colors='red')
        ax.tick_params(axis="x", which="major",
                       labelsize=10, labelrotation=45)

        # aqueous
        ax2 = ax.twinx()
        ax2.plot(well_ts[iwell]["times"],
                 np.abs(well_ts[iwell]["aqueous"]),
                 color="blue")
        ax2.set_ylim(-0.1*max_aqueous, max_aqueous*1.1)
        ax2.tick_params(axis='y', colors='blue')
        ax2.spines['right'].set_color('blue')
        ax2.set_ylabel("Aqueous mass (GPM)",
                       fontsize=10, color="blue")

        # conc
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        ax3.plot(well_ts[iwell]["times"],
                 np.abs(well_ts[iwell]["ctet_conc"]),
                 color="orange")
        ax3.set_ylim(-0.1*max_conc, max_conc*1.1)
        ax3.tick_params(axis='y', colors='orange')
        ax3.spines['right'].set_color('orange')
        ax3.set_ylabel("CTET concentration (ug/L)",
                       fontsize=10, color="orange")
    plt.subplots_adjust(top=0.97, bottom=0.04, left=0.07,
                        right=0.9, wspace=0.55, hspace=0.4)
    fig.set_size_inches(30, 15)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)

    nwell = len(selected_wells)
    ncol = 5
    nrow = int(np.ceil(nwell/ncol))
    imgfile = img_dir+"well_ts_clean.png"
    fig, axs = plt.subplots(nrow, ncol)

    for i, ax in enumerate(fig.axes[nwell:]):
        ax.set_axis_off()

    for i, ax in enumerate(fig.axes[0: nwell]):
        iwell = selected_wells[i]
        ax.plot(
            well_ts[iwell]["times"][np.abs(well_ts[iwell]["aqueous"]) >= 50],
            well_ts[iwell]["ctet_mass"][np.abs(
                well_ts[iwell]["aqueous"]) >= 50],
            color='red',
            zorder=100,
            label=iwell)
        if iwell in selected_wells:
            ax.set_title(iwell, fontsize=15, color="red", fontweight="bold")
        else:
            ax.set_title(iwell, fontsize=15)

        ax.set_ylim(-0.1*max_mass, max_mass*1.1)
        ax.set_ylabel("CTET mass (kg)", fontsize=10, color="red")
        ax.spines['left'].set_color('red')
        ax.tick_params(axis='y', colors='red')
        ax.tick_params(axis="x", which="major",
                       labelsize=10, labelrotation=45)

        # # aqueous
        ax2 = ax.twinx()
        ax2.plot(well_ts[iwell]["times"][np.abs(well_ts[iwell]["aqueous"]) >= 50],
                 np.abs(well_ts[iwell]["aqueous"]
                        [np.abs(well_ts[iwell]["aqueous"]) >= 50]),
                 color="blue")
        ax2.set_ylim(-0.1*max_aqueous, max_aqueous*1.1)
        ax2.tick_params(axis='y', colors='blue')
        ax2.spines['right'].set_color('blue')
        ax2.set_ylabel("Aqueous mass (GPM)",
                       fontsize=10, color="blue")

        # conc
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        ax3.plot(well_ts[iwell]["times"][np.abs(well_ts[iwell]["aqueous"]) >= 50],
                 np.abs(well_ts[iwell]["ctet_conc"])[
            np.abs(well_ts[iwell]["aqueous"]) >= 50],
            color="orange")
        ax3.set_ylim(-0.1*max_conc, max_conc*1.1)
        ax3.tick_params(axis='y', colors='orange')
        ax3.spines['right'].set_color('orange')
        ax3.set_ylabel("CTET concentration (ug/L)",
                       fontsize=10, color="orange")
    plt.subplots_adjust(top=0.97, bottom=0.04, left=0.07,
                        right=0.9, wspace=0.55, hspace=0.4)
    fig.set_size_inches(30, 15)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


def plot_well_map():

    wells = list(well_ts.keys())
    wells = [x for x in wells if "YE" in well_ts[x]["id"]]
    nwell = len(wells)

    west_corner = 566000
    east_corner = 569000
    south_corner = 133500
    north_corner = 138500

    plot_coords = {}
    for iwell in wells:
        well_index = well_info["well"].index(iwell)
        plot_coords[iwell] = [(float(well_info["x"][well_index])-west_corner) /
                              (east_corner-west_corner),
                              (float(well_info["y"][well_index])-south_corner) /
                              (north_corner-south_corner)]

    max_mass = 0
    max_aqueous = 0
    max_conc = 0
    for iwell in wells:
        max_mass = max(max_mass, np.max(well_ts[iwell]["ctet_mass"]))
        max_aqueous = max(max_aqueous, np.max(
            np.abs(well_ts[iwell]["aqueous"])))
        max_conc = max(max_conc, np.max(
            np.abs(well_ts[iwell]["ctet_conc"])))

    imgfile = img_dir+"map_well_ts.png"
    fig = plt.figure()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax = plt.subplot(111)
    for iwell in wells:
        ax = plt.axes([plot_coords[iwell][0],
                       plot_coords[iwell][1],
                       0.08, 0.04])
        # mass
        ax.plot(
            well_ts[iwell]["times"],
            well_ts[iwell]["ctet_mass"],
            color='red',
            zorder=100,
            label=iwell)
        ax.set_title(iwell, fontsize=15)
        ax.set_ylim(-0.1*max_mass, max_mass*1.1)
#        ax.set_ylabel("CTET mass (kg)", fontsize=10, color="red")
        ax.set_ylabel("")
        ax.spines['left'].set_color('red')
        ax.tick_params(axis='y', colors='red')
        ax.tick_params(axis="x", which="major",
                       labelsize=10, labelrotation=45)
        ax.set_xticks([])
        ax.set_yticks([])

        # aqueous
        ax2 = ax.twinx()
        ax2.plot(well_ts[iwell]["times"],
                 np.abs(well_ts[iwell]["aqueous"]),
                 color="blue")
        ax2.set_ylim(-0.1*max_aqueous, max_aqueous*1.1)
        ax2.tick_params(axis='y', colors='blue')
        ax2.spines['right'].set_color('blue')
        ax2.set_ylabel("")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # conc
        ax3 = ax.twinx()
        ax3.plot(well_ts[iwell]["times"],
                 np.abs(well_ts[iwell]["ctet_conc"]),
                 color="orange")
        ax3.set_ylim(-0.1*max_conc, max_conc*1.1)
        ax3.tick_params(axis='y', colors='orange')
        ax3.set_ylabel("")
        ax3.set_xticks([])
        ax3.set_yticks([])

    fig.set_size_inches(30, 30*(north_corner-south_corner) /
                        (east_corner-west_corner))
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)


data_dir = "/rcfs/projects/ml_pt/processed_data/"
well_info_file = data_dir+"well_info.joblib"
well_ts_file = data_dir+"well_ts.joblib"
img_dir = "/people/song884/all_figures/optimize_pt/bulk/"
p2r_dir = "/rcfs/projects/ml_pt/processed_p2r/"
plot_file = "/rcfs/projects/optimize_pt/OP_2/test30/1_1/plot.3021"

well_info = joblib.load(well_info_file)
well_ts = joblib.load(well_ts_file)


selected_wells = ['299-W15-225',
                  '299-W14-20',
                  '299-W14-73',
                  '299-W14-74',
                  '299-W12-2',
                  '299-W11-50',
                  '299-W11-90',
                  '299-W11-96',
                  '299-W17-3',
                  '299-W17-2',
                  #                  '299-W19-111',
                  '299-W11-49',
                  '299-W11-97',
                  '299-W6-15',
                  '299-W14-21',
                  '299-W11-92',
                  '299-W5-1',
                  '299-W12-3',
                  '299-W12-4',
                  '299-W14-22']


snapshot = extract_snapshot_data(plot_file)
# plot_ts()
# plot_relation()
# plot_relation_all()
# plot_cumulative_all()
# plot annual data


def plot_snapshot():
    snapshot_theta = snapshot["varis"]['aqueous moisture content']["value"]
    snapshot_conc = snapshot["varis"]['aqueous plume concentration']["value"]
    snapshot_volume = snapshot["varis"]['node volume']["value"]
    snapshot_mass = snapshot_theta*snapshot_conc*snapshot_volume/1000/1000  # kg

    # total mass
    wells = list(well_ts.keys())
    wells = [x for x in wells if "YE" in well_ts[x]["id"]]
    total_well_mass = np.sum(
        [np.sum(np.array(well_ts[iwell]["ctet_mass"])) for iwell in wells])
    total_well_aqueous = np.sum(
        [np.nansum(np.array(well_ts[iwell]["aqueous"])) for iwell in wells])

    total_mass = np.sum(snapshot_mass)
    total_aqueous = np.sum((snapshot_theta*snapshot_volume)[snapshot_conc > 0])

    # conc distribution
    conc_all = [np.array(well_ts[iwell]["ctet_conc"]) for iwell in wells]
    conc_all = np.array([x for y in conc_all for x in y])
    aqueous_all = [np.array(well_ts[iwell]["aqueous"]) for iwell in wells]
    aqueous_all = np.abs(np.array([x for y in aqueous_all for x in y]))
    conc_all = conc_all[aqueous_all > 0]
    aqueous_all = aqueous_all[aqueous_all > 0]

    clean_up_level = 3.4
    ucl_cutoff = clean_up_level*0.1
    ucl_percentile = 95
    compute_upper_confidence_limit(
        snapshot, ucl_cutoff, ucl_percentile)

    imgfile = img_dir+"snapshot_hist.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    threshold = 0
    data = snapshot_conc[snapshot_conc > threshold]
    weights = (snapshot_volume*snapshot_theta)[snapshot_conc > threshold]
    bins = np.linspace(0, 2000, 100)
    ax.hist(data,
            weights=weights,
            bins=bins,
            density=True)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 0.03)
    ax.set_xlabel("CTET concentration (ug/L)")
    ax.set_ylabel("pdf")

    ax.hist(conc_all,
            weights=aqueous_all,
            fc="None",
            ec="black",
            bins=bins,
            density=True)
#    ax.set_ylim(0, 0.03*)

#    ax.set_yscale("log")
#    ax.set_ylim(0, 0.01)
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.12,
                        right=0.97, wspace=0.55, hspace=0.4)
    fig.set_size_inches(6, 6)
    fig.savefig(imgfile, bbox_inches=0)
    plt.close(fig)
