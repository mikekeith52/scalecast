from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode

theta = {
    'theta':[0.5,1,1.5,2,2.5,3],
    'model_mode':[
        ModelMode.ADDITIVE,
        ModelMode.MULTIPLICATIVE
    ],
    'season_mode':[
        SeasonalityMode.MULTIPLICATIVE,
        SeasonalityMode.ADDITIVE
    ],
    'trend_mode':[
        TrendMode.EXPONENTIAL,
        TrendMode.LINEAR
    ],
}