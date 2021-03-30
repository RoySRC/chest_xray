import torch

class model_base:
    '''
    This is the model base class used as a parent of any model. This baseclass defines a set of functions
    that are generally implemented by the model class. This is an effort at clean coding to reduce the
    number of repeated functions.
    '''

    def __init__(self):
        self.iterations = 0

    def update_iterations(self):
        '''
        Increment the number of iterations by 1
        :return: The new updated iteration count
        '''
        self.iterations += 1
        return self.iterations

    def save(self, path='./'):
        '''
        Save the current model state to file
        :param path: Path to save the model, this also includes the filename
        :return: None
        '''
        checkpoint = {
            'epoch': self.iterations,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def load(self, checkpoint_path):
        '''
        Load a saved model from a checkpoint and return the number of epochs it was trained for.
        :param checkpoint_path: The path to where the model is saved
        :return: The number of epocs the model was trained for
        '''
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']

    def plot(self):
        raise NotImplemented('Function for plotting model needs implementation')