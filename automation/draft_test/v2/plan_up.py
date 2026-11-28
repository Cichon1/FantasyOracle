import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import planner as PN
PN.PL["up80"] = PN.make_plan(0.25, value="up80")
PN.PL["mix"] = PN.make_plan(0.25, value="mix")
import tournament as T
_sd = T.season_data
def sd(y, lg):
    D = _sd(y, lg); P = D["P"]; P["mix"] = 0.5*P["exp_season"] + 0.5*P["up80"].where(P["up80"]>0, P["exp_season"]); return D
T.season_data = sd; PN.T.season_data = sd
PN.run(sys.argv[1], [int(x) for x in sys.argv[2:]], [("c_adp",None),("plan","up80"),("plan","mix")], tag="up")
