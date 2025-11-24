
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from sklearn.cluster import KMeans
from mw_plot import MWSkyMap
def function_task1():
    mw = MWSkyMap(
        center =(0,0) * u.deg,
        radius = (180,90) *u.deg,
        grayscale = False,
        background = "CDS/P/Mellinger/color"
    )

    fig, ax = plt. subplots(figsize=(12,7))
    mw.transform(ax)
    plt.show()
    return fig,ax

def function_task2():
    #Andromeda Galaxy (M31)

    g1 = MWSkyMap(
        center ="M31",
        radius = (6000,6000) * u.arcsec,
        background = "Mellinger color optical survey",
    )
    fig1,ax1 = plt.subplots(figsize=(6,6))
    g1.transform(ax1)

    #only necessary image to save for later:
    fig1.savefig("Andromeda",dpi=300)


    # Centaurus A (NGC 5128)
    g2 = MWSkyMap(
        center ="NGC 5128",
        radius = (6000,6000) * u.arcsec,
        background = "CDS/P/Mellinger/color",
    )
    fig2,ax2 = plt.subplots(figsize=(6,6))
    g2.transform(ax2)

    # Triangulum Galaxy (M33)
    g3 = MWSkyMap(
        center ="M33",
        radius = (8800,8800) * u.arcsec,
        background = "CDS/P/Mellinger/color",
    )
    fig3,ax3 = plt.subplots(figsize=(6,6))
    g3.transform(ax3)


    # Large Magellanic Cloud
    g4 = MWSkyMap(
        center ="LMC",
        radius = (6000,6000) * u.arcsec,
        background = "CDS/P/Mellinger/color",
    )
    fig4,ax4 = plt.subplots(figsize=(6,6))
    g4.transform(ax4)

    return(fig1 , ax1, fig2, ax2, fig3, ax3, fig4, ax4)

def figure_to_rgb_array(fig):
    
    #Remove the extra fat
    fig.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
    fig.canvas.draw()

    #Get the dimensions
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8)

    #Reshape and take off alpha
    rgb = rgba.reshape((h, w, 4))[:,:,:3]

    plt.show()

    #normalize to 0-1 as it is easier to work with a small range
    return rgb/255.0




def generate_color_categories(img_array):
    img = img_array.copy()
    h, w, _ = img.shape

    categories = {
        0:{"name":"Stars","rgb_range": [(0.85,1.0), (0.85, 1.0), (0.85, 1.0)]},
        1:{"name":"Red Nebula","rgb_range": [(0.6, 1.0), (0.0, 0.4), (0.0, 0.4)]},
        2:{"name":"Blue Nebula","rgb_range": [(0.0, 0.5), (0.5, 1.0), (0.7, 1.0)]},
        3:{"name":"Dust / Dark lanes","rgb_range": [(0.0, 0.3), (0.0, 0.2), (0.0, 0.2)]},
        4:{"name":"Background","rgb_range": [(0.0, 0.08), (0.0, 0.08), (0.0, 0.08)]},
        
    }

    category_map = np.full((h, w),-1)

    for cat_id, cat in categories.items():
        r_min, r_max = cat["rgb_range"][0]
        g_min, g_max = cat["rgb_range"][1]
        b_min, b_max = cat["rgb_range"][2]

        mask = (
            (img[:,:,0] >= r_min) & (img[:,:,0] <= r_max) &
            (img[:,:,1] >= g_min) & (img[:,:,1] <= g_max) &
            (img[:,:,2] >= b_min) & (img[:,:,2] <= b_max) 
        )
        category_map[mask] = cat_id

    return category_map


def cluster_KMeans_pixels (img_array, category_map, n_clusters=5):
    img = img_array.copy()
    unassigned_mask = (category_map == -1 )
    clustered_map = category_map.copy()

    if np.any(unassigned_mask):
        pixels = img[unassigned_mask]
        kmeans = KMeans(n_clusters=n_clusters, random_state = 42)
        labels = kmeans.fit_predict(pixels)

        clustered_map[unassigned_mask]=labels+10
        return clustered_map, kmeans.cluster_centers_
    else:
        return category_map, None
    

def plot_clusters_overlay(img_array, cluster_map, cluster_centers=None, step=2, alpha=0.8):
    h, w, _ = img_array.shape 
    unique_clusters = np.unique(cluster_map)
    k = len(unique_clusters)

    #Map cluster IDs to indices 
    cluster_id_to_idx = {cid: i for i, cid in enumerate (unique_clusters)}

    #Build overlaying colors
    overlay_colors = np.zeros((k,3))
    for i, cl in enumerate (unique_clusters):
        if cluster_centers is not None and cl >= 10:
            overlay_colors[i] = np.clip(cluster_centers[cl - 10], 0.0, 1.0)
        else:
            overlay_colors[i]= {
                0: np.array([1.0, 1.0, 1.0]), #Stars color
                1: np.array([1.0, 0.4, 0.4]), #Red Nebula color
                2: np.array([0.4, 0.6, 1.0]), #Blue Nebula
                3: np.array([0.0, 0.0, 0.0]), #Dust color
                4: np.array([0.0, 0.0, 0.0])  #Background color as dust
            }.get(cl, img_array[cluster_map == cl].mean(axis=0))

        #Build a grid to present the clusters overlay
        yy, xx = np.mgrid[0:h, 0:w]
        xx_s = xx[::step, ::step]
        yy_s = yy[::step, ::step]
        clusters_s = cluster_map[::step, ::step]

        indices_s = np.vectorize(lambda x:cluster_id_to_idx[x])(clusters_s)
        colors_s = overlay_colors[indices_s.flatten()]

        #Plot
        plt.figure(figsize = (8,8))
        plt.imshow(img_array)
        plt.scatter(xx_s, yy_s, c=colors_s,s=1, alpha=alpha)
        plt.axis("off")
        plt.show()
