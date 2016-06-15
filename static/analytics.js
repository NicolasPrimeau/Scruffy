$(function () {

    $.get( "analytics/get_scores", function( data ) {

        var scores_preprocess = JSON.parse(data);
        var scores = [];
        for (var i=0; i<scores_preprocess.length; i+=1) {
            scores.push([i, scores_preprocess[i].score]);
        }

        $('#container').highcharts({
            chart: {
                type: 'scatter',
                zoomType: 'xy'
            },
            title: {
                text: 'Scores'
            },
            xAxis: {
                title: {
                    enabled: true,
                    text: 'Time'
                },
                startOnTick: true,
                endOnTick: true,
                showLastLabel: true
            },
            yAxis: {
                title: {
                    text: 'Score'
                }
            },

            plotOptions: {
                scatter: {
                    marker: {
                        radius: 5,
                        states: {
                            hover: {
                                enabled: true,
                                lineColor: 'rgb(100,100,100)'
                            }
                        }
                    },
                    states: {
                        hover: {
                            marker: {
                                enabled: false
                            }
                        }
                    }
                }
            },
            series: [{
                name: 'Data',
                color: 'rgba(223, 83, 83, .5)',
                data: scores

            }]
        });
    });
});
