from shape_match_model_pb import ShapeMatchModel
import time
from utils.sm_utils import *
import scipy.sparse as sp
from utils.misc import robust_lossfun
from utils.vis_util import plot_match, get_cams_and_rotation
from utils.disco_utils import disco_solver, get_solver_opts

## Load data (change dataset accordingly when using files from different datasets)
dataset = "faust" # dt4d_inter, dt4d_intra, smal
filename1 = "datasets/FAUST_r/off/tr_reg_000.off"
filename2 = "datasets/FAUST_r/off/tr_reg_001.off"
shape_opts = {"num_faces": 100} # takes around 60s on M1 Mac Pro, much faster on GPU, full performance of our solver only shows on GPU!


## Load and downsample shapes and compute spidercurve on shape X
VX, FX, vx, fx, vx2VX, VY, FY, vy, fy, vy2VY = shape_loader(filename1, filename2, shape_opts)

## Comptue Features and edge cost matrix
feature_opts = get_feature_opts(dataset)
feat_x, feat_y = get_features(VX, FX, VY, FY, feature_opts)
cost_matrix = np.zeros((len(vx), len(vy)))
for i in range(0, len(vx)):
    diff = feat_y[vy2VY, :] - feat_x[vx2VX[i], :]
    cost_matrix[i, :] = np.sum(to_numpy(robust_lossfun(torch.from_numpy(diff.astype('float64')),
                                                       alpha=torch.tensor(2, dtype=torch.float64),
                                                       scale=torch.tensor(0.3, dtype=torch.float64))), axis=1)

## ++++++++++++++++++++++++++++++++++++++++
## ++++++++ Solve with DISCOMATCH +++++++++
## ++++++++++++++++++++++++++++++++++++++++
smm = ShapeMatchModel(fx, vx, fy, vy)
smm.updateEnergy(cost_matrix, True, False, 1.0)

start_time = time.time()
lower_bound, G, primal_feasible = disco_solver(smm, get_solver_opts())
end_time = time.time()
print(f"Optimisation took {end_time - start_time}s")

## Visualise result
sG = sparse.csr_matrix(G, dtype=np.int8)
point_map = point_map_orig = smm.getPointMatchesFromSolution(sG)
[cam, cams, rotationX, rotationY] = get_cams_and_rotation(dataset)
plot_match(vy, fy, vx, fx, point_map[:, [1, 0]], cam, "", offsetX=[1, 0, 0],
                            rotationShapeX=rotationX, rotationShapeY=rotationY)