# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ._model_evaluation import plot_confusion_matrix, plot_ROC_curve, plot_PR_curve, plot_ROC_and_PR_curves, demo

# this is for "from <package_name>.model_evaluation import *"
__all__ = ["plot_confusion_matrix",
           "plot_ROC_curve",
           "plot_PR_curve",
           "plot_ROC_and_PR_curves",
           "demo"]
