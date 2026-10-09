import numpy as np
from rato.environment.action_handler import default_actions,project_actions
from rato.optimization.constraints import validate_actions

def test_projection(cfg):
    K=cfg['paper']['network']['num_edge_servers'];M=3
    action=default_actions(M,K)
    projection=project_actions(action,np.array([0,1,0]),K)
    assert validate_actions(projection['gamma'],projection['power_coefficients'],projection['routes'],K)
    np.testing.assert_allclose(projection['power_coefficients'].sum(),1)
