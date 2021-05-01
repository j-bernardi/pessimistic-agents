
def main():

    # pick int (from prior? tbc if geometric dist is appropriate)
    # we don't need an infinite number!! Use uniform distribution over finite number of world models!!!
    # 20 lines. 2^20 world models
    # turn the int into a bitstring and then into a list of boundaries.

    # initialize agent
        # pick world models until sum weight = beta

    # run loop

        # find optimal point
        # move to (towards seems redundant) optimal point
        # if optimal point is past the real world model we ded
        # now we're stuck
        # we ask the mentor?
            # sapleable distribution over mentor policies
            # we sample a mentor policy
            # we work out discounted future reward of that mentor policy
        
        # if we ask the mentor and it gives us a point which is beyond a boundary then we get rid of that boundary
        # and re-pick world models until sum weight = beta
    
        # if we don't ask the mentor we end the loop.
    


    pass
    

def viz(n_boundaries=50):

    pass




if __name__ == "__main__":
    main()
