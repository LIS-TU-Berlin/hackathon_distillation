import hackathon_distillation as hack

def test_random_resets():
    B = hack.ExpertBehavior()
    for i in range(10):
        B.reset()
        B.S.C.view(True)

def test_in_BotOp():
    B = hack.ExpertBehavior()
    B.run_with_BotOp()

def test_generate_data():
    B = hack.ExpertBehavior()
    h5 = hack.H5Writer('data_20251029.h5')
    for i in range(10):
        B.reset()
        B.run_with_Sim(h5=h5, verbose=1)
    #B.run_with_Sim(h5="None", verbose=1)
    

if __name__ == "__main__":
    # test_random_resets()
    # test_in_BotOp()
    test_generate_data()
