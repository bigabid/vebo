import { BarChart, TrendingUp, Database, Calendar, Hash, Type, AlertCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import type { InsightsData, ColumnInsight } from '@/lib/api';

interface InsightsPanelProps {
  insights: InsightsData | null;
}

function StatCard({ 
  icon: Icon, 
  title, 
  value, 
  subtitle,
  color = "primary" 
}: {
  icon: any;
  title: string;
  value: string | number;
  subtitle?: string;
  color?: "primary" | "secondary" | "success" | "warning";
}) {
  return (
    <Card className="p-4 bg-gradient-card border-0 shadow-soft">
      <div className="flex items-center gap-3">
        <div className={`p-2 bg-${color}/10 rounded-lg`}>
          <Icon className={`w-5 h-5 text-${color}`} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold text-foreground truncate">{value}</p>
          {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
        </div>
      </div>
    </Card>
  );
}

function ColumnInsightCard({ column }: { column: ColumnInsight }) {
  const getTypeIcon = (type: string) => {
    if (type.includes('string') || type.includes('varchar')) return Type;
    if (type.includes('int') || type.includes('double') || type.includes('decimal')) return Hash;
    if (type.includes('timestamp') || type.includes('date')) return Calendar;
    return Database;
  };

  const getTypeColor = (type: string) => {
    if (type.includes('string') || type.includes('varchar')) return 'secondary';
    if (type.includes('int') || type.includes('double') || type.includes('decimal')) return 'primary';
    if (type.includes('timestamp') || type.includes('date')) return 'success';
    return 'muted';
  };

  const TypeIcon = getTypeIcon(column.type);
  const typeColor = getTypeColor(column.type);
  const nullPercentage = Math.round(column.nullRatio * 100);

  const formatNumber = (n?: number | null) =>
    typeof n === 'number' && isFinite(n) ? n.toLocaleString() : '—';

  return (
    <Card className="p-5 bg-gradient-card border-0 shadow-soft hover:shadow-medium transition-shadow">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TypeIcon className={`w-4 h-4 text-${typeColor}`} />
            <h4 className="font-medium text-foreground">{column.name}</h4>
          </div>
          <div className="flex items-center gap-2">
            {column.valueType && (
              <Badge variant="secondary" className="text-xs capitalize">
                {column.valueType}
              </Badge>
            )}
            <Badge variant="outline" className={`bg-${typeColor}/10 text-${typeColor} border-${typeColor}/30`}>
              {column.type}
            </Badge>
          </div>
        </div>

        {/* Null Ratio */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Null Ratio</span>
            <span className={`font-medium ${nullPercentage > 20 ? 'text-warning' : 'text-success'}`}>
              {nullPercentage}%
            </span>
          </div>
          <Progress 
            value={nullPercentage} 
            className="h-1.5"
          />
        </div>

        <Separator />

        {/* Type-specific insights */}
        {column.numeric && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-foreground flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              Numeric Statistics
            </h5>
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div>
                <span className="text-muted-foreground block">Min</span>
                <span className="font-medium">{formatNumber(column.numeric.min)}</span>
              </div>
              <div>
                <span className="text-muted-foreground block">Avg</span>
                <span className="font-medium">{formatNumber(column.numeric.avg)}</span>
              </div>
              <div>
                <span className="text-muted-foreground block">Max</span>
                <span className="font-medium">{formatNumber(column.numeric.max)}</span>
              </div>
            </div>
          </div>
        )}

        {column.temporal && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-foreground flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              Time Range
            </h5>
            <div className="text-sm space-y-1">
              <div>
                <span className="text-muted-foreground">From: </span>
                <span className="font-medium">{new Date(column.temporal.min).toLocaleDateString()}</span>
              </div>
              <div>
                <span className="text-muted-foreground">To: </span>
                <span className="font-medium">{new Date(column.temporal.max).toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        )}

        {column.topValues && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-foreground flex items-center gap-1">
              <BarChart className="w-3 h-3" />
              Top Values
            </h5>
            <div className="space-y-1">
              {column.topValues.slice(0, 3).map((item, index) => (
                <div key={index} className="flex items-center justify-between text-sm">
                  <span className="text-foreground truncate flex-1 mr-2">{item.value}</span>
                  <span className="text-muted-foreground font-mono">
                    {item.count.toLocaleString()}
                  </span>
                </div>
              ))}
              {column.topValues.length > 3 && (
                <div className="text-xs text-muted-foreground">
                  +{column.topValues.length - 3} more values
                </div>
              )}
            </div>
          </div>
        )}

        {/* Additional basic details */}
        {column.basic && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-foreground">Details</h5>
            <div className="grid grid-cols-2 gap-3 text-sm">
              {typeof column.basic.uniqueCount !== 'undefined' && (
                <div>
                  <span className="text-muted-foreground block">Unique Count</span>
                  <span className="font-medium">{formatNumber(column.basic.uniqueCount)}</span>
                </div>
              )}
              {typeof column.basic.uniqueRatio !== 'undefined' && (
                <div>
                  <span className="text-muted-foreground block">Unique Ratio</span>
                  <span className="font-medium">{Math.round((column.basic.uniqueRatio || 0) * 100)}%</span>
                </div>
              )}
              {typeof column.basic.duplicateCount !== 'undefined' && (
                <div>
                  <span className="text-muted-foreground block">Duplicate Count</span>
                  <span className="font-medium">{formatNumber(column.basic.duplicateCount)}</span>
                </div>
              )}
              {typeof column.basic.duplicateRatio !== 'undefined' && (
                <div>
                  <span className="text-muted-foreground block">Duplicate Ratio</span>
                  <span className="font-medium">{Math.round((column.basic.duplicateRatio || 0) * 100)}%</span>
                </div>
              )}
              {typeof column.basic.nullCount !== 'undefined' && (
                <div>
                  <span className="text-muted-foreground block">Null Count</span>
                  <span className="font-medium">{formatNumber(column.basic.nullCount)}</span>
                </div>
              )}
              {typeof column.basic.mostCommonValue !== 'undefined' && (
                <div className="col-span-2">
                  <span className="text-muted-foreground block">Most Common</span>
                  <span className="font-medium break-words">{String(column.basic.mostCommonValue ?? '—')}</span>
                  {typeof column.basic.mostCommonFrequency !== 'undefined' && (
                    <span className="ml-2 text-xs text-muted-foreground">({formatNumber(column.basic.mostCommonFrequency)})</span>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

export function InsightsPanel({ insights }: InsightsPanelProps) {
  if (!insights) {
    return (
      <Card className="p-8 bg-gradient-card border-0 shadow-soft">
        <div className="text-center space-y-3">
          <div className="w-16 h-16 bg-muted/30 rounded-full flex items-center justify-center mx-auto">
            <BarChart className="w-8 h-8 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">No Insights Yet</h3>
            <p className="text-sm text-muted-foreground">
              Select a table and click Execute to generate insights
            </p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6 bg-gradient-primary border-0 shadow-large text-primary-foreground">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Insights for {insights.table}</h2>
            <div className="flex items-center gap-4 text-sm opacity-90">
              {Object.keys(insights.appliedFilters).length > 0 && (
                <span>
                  Filtered by: {Object.entries(insights.appliedFilters)
                    .map(([key, values]) => `${key} (${values.length})`)
                    .join(', ')}
                </span>
              )}
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">{insights.rowCount.toLocaleString()}</div>
            <div className="text-sm opacity-90">Total Rows</div>
          </div>
        </div>
      </Card>

      {/* Primary Key Candidates */}
      {insights.primaryKeys && insights.primaryKeys.length > 0 && (
        <Card className="p-5 bg-gradient-card border-0 shadow-soft">
          <div className="mb-3">
            <h3 className="text-lg font-semibold text-foreground">Primary key candidates</h3>
            <p className="text-sm text-muted-foreground">Ranked by confidence, uniqueness and completeness</p>
          </div>
          <div className="space-y-2">
            {insights.primaryKeys.slice(0, 5).map((ck, idx) => (
              <div key={idx} className="flex items-center justify-between text-sm bg-background border border-border rounded-md px-3 py-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-medium truncate">{ck.columns.join(' + ')}</span>
                  {ck.noNulls && (
                    <Badge variant="outline" className="text-xs">no nulls</Badge>
                  )}
                  {typeof ck.confidence === 'number' && (
                    <Badge variant="secondary" className="text-xs">conf {Math.round(ck.confidence * 100)}%</Badge>
                  )}
                </div>
                <div className="text-right shrink-0">
                  <span className="font-semibold">{Math.round(ck.uniqueness * 100)}%</span>
                </div>
              </div>
            ))}
          </div>
          {insights.primaryKeys[0].reason && (
            <div className="mt-3 text-xs text-muted-foreground">
              Reason: {insights.primaryKeys[0].reason}
            </div>
          )}
        </Card>
      )}

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          icon={Database}
          title="Total Rows"
          value={insights.rowCount.toLocaleString()}
          subtitle="Records analyzed"
          color="primary"
        />
        <StatCard
          icon={BarChart}
          title="Columns"
          value={insights.columns.length}
          subtitle="Data attributes"
          color="secondary"
        />
        <StatCard
          icon={TrendingUp}
          title="Partitions"
          value={insights.partitionSummary.selectedCount}
          subtitle={`of ${insights.partitionSummary.totalDistinct} total`}
          color="success"
        />
      </div>

      {/* Column Analysis */}
      <div>
        <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
          <Type className="w-5 h-5" />
          Column Analysis
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {insights.columns.map((column, index) => (
            <ColumnInsightCard key={index} column={column} />
          ))}
        </div>
      </div>
    </div>
  );
}